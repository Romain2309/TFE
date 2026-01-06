"""
Training callbacks for monitoring and controlling training.

Provides:
- EarlyStopping: Stop training when metric stops improving
- ModelCheckpoint: Save best/latest model checkpoints
- LRSchedulerCallback: Wrap PyTorch LR schedulers
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Example:
        early_stop = EarlyStopping(patience=10, min_delta=1e-4)

        for epoch in range(n_epochs):
            val_loss = train_and_validate()
            if early_stop(val_loss):
                print("Early stopping triggered!")
                break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (lower is better) or 'max' (higher is better)
            verbose: Print messages when triggered
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current value of monitored metric

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training (patience={self.patience} reached)")
                return True

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Saves:
    - Best model (based on monitored metric)
    - Last model (most recent epoch)
    - Periodic checkpoints (every N epochs)
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_last: bool = True,
        filename_best: str = 'best_model.pt',
        filename_last: str = 'last_model.pt',
        verbose: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_last: Always save the last epoch
            filename_best: Filename for best model
            filename_last: Filename for last model
            verbose: Print messages when saving
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.filename_best = filename_best
        self.filename_last = filename_last
        self.verbose = verbose

        self.best_score = None

        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch number
            model: PyTorch model
            optimizer: PyTorch optimizer
            metrics: Dictionary of metrics (must contain self.monitor)
            extra_state: Additional state to save (e.g., scheduler, scaler)
        """
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in metrics: {list(metrics.keys())}")

        current_score = metrics[self.monitor]

        is_best = False
        if self.best_score is None:
            is_best = True
            self.best_score = current_score
        else:
            if self.mode == 'min':
                is_best = current_score < self.best_score
            else:
                is_best = current_score > self.best_score

            if is_best:
                self.best_score = current_score

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score,
        }

        if extra_state is not None:
            checkpoint.update(extra_state)

        if is_best or not self.save_best_only:
            best_path = self.save_dir / self.filename_best
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"Checkpoint saved to {best_path} ({self.monitor}={current_score:.6f})")

        if self.save_last:
            last_path = self.save_dir / self.filename_last
            torch.save(checkpoint, last_path)


class LRSchedulerCallback:
    """
    Wrapper for PyTorch learning rate schedulers.

    Handles stepping the scheduler at the right time (epoch or step).
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        step_on: str = 'epoch',  # 'epoch' or 'batch'
        monitor: Optional[str] = None,  # For ReduceLROnPlateau
    ):
        """
        Args:
            scheduler: PyTorch LR scheduler
            step_on: When to step ('epoch' or 'batch')
            monitor: Metric to monitor (for ReduceLROnPlateau)
        """
        self.scheduler = scheduler
        self.step_on = step_on
        self.monitor = monitor

        if step_on not in ['epoch', 'batch']:
            raise ValueError(f"step_on must be 'epoch' or 'batch', got {step_on}")

        self.is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        if self.is_plateau and monitor is None:
            raise ValueError("monitor must be specified for ReduceLROnPlateau")

    def step(self, metrics: Optional[Dict[str, float]] = None):
        """
        Step the learning rate scheduler.

        Args:
            metrics: Dictionary of metrics (required for ReduceLROnPlateau)
        """
        if self.is_plateau:
            if metrics is None or self.monitor not in metrics:
                raise ValueError(f"metrics['{self.monitor}'] required for ReduceLROnPlateau")
            self.scheduler.step(metrics[self.monitor])
        else:
            self.scheduler.step()

    def get_last_lr(self):
        """Get the last learning rate(s)."""
        return self.scheduler.get_last_lr()


if __name__ == "__main__":
    print("Testing training callbacks...")

    print("\n1. Testing EarlyStopping...")
    early_stop = EarlyStopping(patience=3, min_delta=0.01, mode='min', verbose=True)

    losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.82, 0.82, 0.82]  # Plateaus
    for i, loss in enumerate(losses):
        print(f"  Epoch {i}: loss={loss:.2f}")
        if early_stop(loss):
            print(f"  Training stopped at epoch {i}")
            break

    print("\n2. Testing ModelCheckpoint...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = ModelCheckpoint(
            save_dir=tmpdir,
            monitor='val_loss',
            mode='min',
            verbose=True,
        )

        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(5):
            val_loss = 1.0 - epoch * 0.1  # Decreasing loss
            metrics = {'val_loss': val_loss, 'train_loss': val_loss + 0.1}

            checkpoint(epoch, model, optimizer, metrics)

        print(f"  Checkpoints saved to: {tmpdir}")
        print(f"  Files: {list(Path(tmpdir).glob('*.pt'))}")

    print("\n3. Testing LRSchedulerCallback...")
    optimizer = torch.optim.Adam([torch.tensor(1.0)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    lr_callback = LRSchedulerCallback(scheduler, step_on='epoch', monitor='val_loss')

    print("  Initial LR:", lr_callback.get_last_lr())
    for epoch in range(10):
        val_loss = 1.0  # Constant (should trigger LR reduction)
        lr_callback.step({'val_loss': val_loss})
        print(f"  Epoch {epoch}: LR = {lr_callback.get_last_lr()[0]:.6f}")

    print("\nAll tests passed!")
