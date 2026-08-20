"""
Unified training loop for all model types.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None
import time
from typing import Dict, Optional, Any
from tqdm import tqdm
import numpy as np

from .config import TrainingConfig
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from ..models.losses import AngularLoss, CosineLoss, MultiTaskLoss, angular_separation_degrees
from ..evaluation.metrics import (
    compute_angular_errors,
    angular_error_statistics,
    searched_area_batch,
    classification_accuracy,
    compute_time_delay_metrics,
)
from ..utils.coordinates import get_healpix_npix


class Trainer:
    """
    Unified trainer for all 4 model types.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        Args:
            model: PyTorch model (one of 4 types)
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
            test_loader: Test DataLoader (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        self.model.to(self.device)

        self.criterion = self._create_criterion()

        self.optimizer = self._create_optimizer()

        self.scheduler = self._create_scheduler()

        self.early_stopping = None
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
                mode='min' if 'loss' in config.monitor_metric else 'max',
                verbose=config.verbose,
            )

        self.checkpoint = ModelCheckpoint(
            save_dir=config.output_dir,
            monitor=config.monitor_metric,
            mode='min' if 'loss' in config.monitor_metric else 'max',
            save_best_only=config.save_best_only,
            save_last=config.save_last,
            verbose=config.verbose,
        )

        self.writer = None
        if config.use_tensorboard:
            if not HAS_TENSORBOARD:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                print("Continuing without TensorBoard logging...")
            else:
                log_dir = Path(config.output_dir) / 'logs'
                self.writer = SummaryWriter(log_dir=str(log_dir))

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = None
        self.history = {'train': [], 'val': []}

        if config.verbose:
            print(f"\nTrainer initialized:")
            print(f"  Model type: {config.model_type}")
            print(f"  Device: {self.device}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Output directory: {config.output_dir}")

    def _create_criterion(self):
        """Create loss function based on model type."""
        if self.config.model_type == 'classifier':
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        elif self.config.model_type == 'probmap':
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        elif self.config.model_type == 'regressor':
            if self.config.loss_type == 'angular':
                return AngularLoss()
            elif self.config.loss_type == 'cosine':
                return CosineLoss()
            else:
                raise ValueError(f"Unknown loss_type: {self.config.loss_type}")
        elif self.config.model_type == 'multitask':
            return MultiTaskLoss(
                tau_weight=self.config.tau_weight,
                sky_loss_type=self.config.loss_type,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _create_optimizer(self):
        """Create optimizer."""
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type is None:
            return None
        elif self.config.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            )
            return LRSchedulerCallback(scheduler, step_on='epoch', monitor='val_loss')
        elif self.config.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_factor,
            )
            return LRSchedulerCallback(scheduler, step_on='epoch')
        elif self.config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs,
            )
            return LRSchedulerCallback(scheduler, step_on='epoch')
        elif self.config.scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.n_epochs,
                steps_per_epoch=len(self.train_loader),
            )
            return LRSchedulerCallback(scheduler, step_on='batch')
        else:
            raise ValueError(f"Unknown scheduler_type: {self.config.scheduler_type}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.n_epochs}")

        for batch_idx, batch in enumerate(pbar):
            strain = batch['strain'].to(self.device)
            physics_features = batch['physics_features'].to(self.device)
            xyz_target = batch['xyz'].to(self.device)

            self.optimizer.zero_grad()

            if self.config.model_type in ['classifier', 'probmap']:
                healpix_target = batch['healpix_label'].to(self.device)
                logits = self.model(strain, physics_features)
                loss = self.criterion(logits, healpix_target)

            elif self.config.model_type == 'regressor':
                xyz_pred = self.model(strain, physics_features)
                loss = self.criterion(xyz_pred, xyz_target)

            elif self.config.model_type == 'multitask':
                tau_target = physics_features[:, :2]  # tau_HL, tau_HV
                xyz_pred, tau_pred = self.model(strain, physics_features)
                loss = self.criterion(xyz_pred, tau_pred, xyz_target, tau_target)

            loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            if self.scheduler is not None and self.scheduler.step_on == 'batch':
                self.scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix({'loss': running_loss / n_batches})

            if self.writer is not None and self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)

        avg_loss = running_loss / n_batches

        return {'train_loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        running_loss = 0.0
        n_batches = 0

        all_predictions = []
        all_targets = []
        all_xyz_pred = []
        all_xyz_target = []
        all_tau_pred = []
        all_tau_target = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            strain = batch['strain'].to(self.device)
            physics_features = batch['physics_features'].to(self.device)
            xyz_target = batch['xyz'].to(self.device)

            if self.config.model_type in ['classifier', 'probmap']:
                healpix_target = batch['healpix_label'].to(self.device)
                logits = self.model(strain, physics_features)
                loss = self.criterion(logits, healpix_target)

                if self.config.model_type == 'classifier':
                    pred_pixels = torch.argmax(logits, dim=1)
                    all_predictions.append(pred_pixels.cpu().numpy())
                else:  # probmap
                    probs = torch.softmax(logits, dim=1)
                    all_predictions.append(probs.cpu().numpy())

                all_targets.append(healpix_target.cpu().numpy())

            elif self.config.model_type == 'regressor':
                xyz_pred = self.model(strain, physics_features)
                loss = self.criterion(xyz_pred, xyz_target)

                all_xyz_pred.append(xyz_pred.cpu().numpy())
                all_xyz_target.append(xyz_target.cpu().numpy())

            elif self.config.model_type == 'multitask':
                tau_target = physics_features[:, :2]
                xyz_pred, tau_pred = self.model(strain, physics_features)
                loss = self.criterion(xyz_pred, tau_pred, xyz_target, tau_target)

                all_xyz_pred.append(xyz_pred.cpu().numpy())
                all_xyz_target.append(xyz_target.cpu().numpy())
                all_tau_pred.append(tau_pred.cpu().numpy())
                all_tau_target.append(tau_target.cpu().numpy())

            running_loss += loss.item()
            n_batches += 1

        metrics = {'val_loss': running_loss / n_batches}

        if self.config.model_type == 'classifier':
            pred_pixels = np.concatenate(all_predictions)
            true_pixels = np.concatenate(all_targets)
            metrics['val_accuracy'] = classification_accuracy(pred_pixels, true_pixels)

        elif self.config.model_type == 'regressor':
            xyz_pred = np.concatenate(all_xyz_pred)
            xyz_target = np.concatenate(all_xyz_target)
            errors = compute_angular_errors(xyz_pred, xyz_target, degrees=True)
            error_stats = angular_error_statistics(errors)
            metrics['val_median_error'] = error_stats['median']
            metrics['val_mean_error'] = error_stats['mean']
            metrics['val_p90_error'] = error_stats['p90']

        elif self.config.model_type == 'multitask':
            xyz_pred = np.concatenate(all_xyz_pred)
            xyz_target = np.concatenate(all_xyz_target)
            tau_pred = np.concatenate(all_tau_pred)
            tau_target = np.concatenate(all_tau_target)

            errors = compute_angular_errors(xyz_pred, xyz_target, degrees=True)
            error_stats = angular_error_statistics(errors)
            metrics['val_median_error'] = error_stats['median']
            metrics['val_mean_error'] = error_stats['mean']

            tau_metrics = compute_time_delay_metrics(tau_pred, tau_target)
            metrics['val_tau_mae_ms'] = tau_metrics['mae_ms']

        elif self.config.model_type == 'probmap':
            probs = np.concatenate(all_predictions)
            true_pixels = np.concatenate(all_targets)

            n_samples = min(100, len(true_pixels))
            sa_results = searched_area_batch(
                probs[:n_samples],
                true_pixels[:n_samples],
                nside=self.config.healpix_nside,
            )
            metrics['val_searched_area'] = float(np.median(sa_results['searched_area_deg2']))
            metrics['val_true_in_90'] = float(np.mean(sa_results['true_in_90']))

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Dictionary with training history and final metrics
        """
        if self.config.verbose:
            print(f"\nStarting training for {self.config.n_epochs} epochs...")
            print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(self.config.n_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()

            val_metrics = self.validate()

            metrics = {**train_metrics, **val_metrics}

            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

            if self.config.verbose:
                print(f"\nEpoch {epoch+1}/{self.config.n_epochs}:")
                for key, val in metrics.items():
                    print(f"  {key}: {val:.6f}")

            if self.writer is not None:
                for key, val in metrics.items():
                    self.writer.add_scalar(f'epoch/{key}', val, epoch)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)

            if self.scheduler is not None and self.scheduler.step_on == 'epoch':
                self.scheduler.step(metrics)

            self.checkpoint(epoch, self.model, self.optimizer, metrics)

            if self.early_stopping is not None:
                monitor_value = metrics[self.config.monitor_metric]
                if self.early_stopping(monitor_value):
                    if self.config.verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break

        elapsed = time.time() - start_time

        if self.config.verbose:
            print(f"\nTraining complete in {elapsed/60:.2f} minutes")

        if self.writer is not None:
            self.writer.close()

        return {
            'history': self.history,
            'final_metrics': metrics,
            'epochs_trained': self.current_epoch + 1,
            'time_elapsed': elapsed,
        }


"""
if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use this in conjunction with scripts/train.py for actual training.")
"""
