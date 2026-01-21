#!/usr/bin/env python3

import argparse
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gw_skylocalization.training.config import TrainingConfig
from gw_skylocalization.training.trainer import Trainer
from gw_skylocalization.data.loaders import create_dataloaders
from gw_skylocalization.data.hdf5_loader import create_hdf5_dataloaders
from gw_skylocalization.models.networks import SkyClassifier, SkyRegressor, SkyMultiTask, SkyProbabilityMap
from gw_skylocalization.utils.io import save_results


def parse_args():
    parser = argparse.ArgumentParser(description='Train gravitational wave sky localization models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file (overrides other arguments)')
    parser.add_argument('--model', type=str, default='regressor', choices=['classifier', 'regressor', 'multitask', 'probmap'], help='Model type to train')
    parser.add_argument('--data-dir', type=str, default='/data/stu_bonhomme/tfe/benchmark_same_small', help='Path to data directory (for .npy files)')
    parser.add_argument('--hdf5-path', type=str, default=None, help='Path to preprocessed HDF5 file (faster! Use scripts/preprocess_dataset.py to create)')
    parser.add_argument('--in-memory', action='store_true', help='Load HDF5 dataset into RAM (requires ~15-20GB for 30K events)')
    parser.add_argument('--max-events', type=int, default=None, help='Maximum number of events to use (None = use all)')
    parser.add_argument('--train-frac', type=float, default=0.7, help='Fraction of data for training')
    parser.add_argument('--val-frac', type=float, default=0.15, help='Fraction of data for validation')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--no-physics-features', action='store_true', help='Disable physics features (use only raw strains)')
    parser.add_argument('--healpix-nside', type=int, default=8, help='HEALPix nside for classifier/probmap')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step', 'onecycle', 'none'], help='Learning rate scheduler')
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    return parser.parse_args()

def create_model(config: TrainingConfig):
    model_kwargs = {
        'use_physics_features': config.use_physics_features,
        'strain_feature_dim': config.strain_feature_dim,
        'physics_feature_dim': config.physics_feature_dim,
    }

    if config.model_type == 'classifier':
        model = SkyClassifier(n_pixels=config.n_pixels, **model_kwargs)
    elif config.model_type == 'regressor':
        model = SkyRegressor(**model_kwargs)
    elif config.model_type == 'multitask':
        model = SkyMultiTask(**model_kwargs)
    elif config.model_type == 'probmap':
        model = SkyProbabilityMap(n_pixels=config.n_pixels, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    return model


def main():
    args = parse_args()

    if args.config is not None:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig(
            model_type=args.model,
            data_dir=args.data_dir,
            max_events=args.max_events,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=1.0 - args.train_frac - args.val_frac,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_physics_features=not args.no_physics_features,
            healpix_nside=args.healpix_nside,
            optimizer=args.optimizer,
            scheduler_type=None if args.scheduler == 'none' else args.scheduler,
            early_stopping=not args.no_early_stopping,
            patience=args.patience,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            use_tensorboard=not args.no_tensorboard,
            verbose=args.verbose,
            device=args.device,
            seed=args.seed,
        )

    if config.verbose:
        print("\n" + "="*80)
        print("ROMAIN - Gravitational Wave Sky Localization")
        print("="*80)
        print(config)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config.save(output_path / 'config.json')

    if config.verbose:
        print("Creating data loaders...")

    use_hdf5 = hasattr(args, 'hdf5_path') and args.hdf5_path is not None
    in_memory = hasattr(args, 'in_memory') and args.in_memory

    if use_hdf5:
        if config.verbose:
            print(f"  Loading from HDF5: {args.hdf5_path}")
            if in_memory:
                print("  Loading entire dataset into RAM...")

        train_loader, val_loader, test_loader = create_hdf5_dataloaders(hdf5_path=args.hdf5_path, batch_size=config.batch_size, train_frac=config.train_frac,
                                                                        val_frac=config.val_frac, test_frac=config.test_frac, num_workers=config.num_workers, 
                                                                        pin_memory=config.pin_memory, seed=config.seed, in_memory=in_memory, verbose=config.verbose)
    else:
        if config.verbose:
            print(f"  Loading from .npy files: {config.data_dir}")

        train_loader, val_loader, test_loader = create_dataloaders(data_dir=config.data_dir, batch_size=config.batch_size, target_length=config.target_length, 
                                                                   bandpass=(config.bandpass_low, config.bandpass_high), normalize=config.normalize, use_healpix=True,
                                                                   healpix_nside=config.healpix_nside, train_frac=config.train_frac, val_frac=config.val_frac, 
                                                                   test_frac=config.test_frac, max_events=config.max_events, num_workers=config.num_workers, 
                                                                   cache_train=config.cache_train, seed=config.seed, verbose=config.verbose)

    if config.verbose and not use_hdf5:
        print(f"  Train: {len(train_loader.dataset)} events ({len(train_loader)} batches)")
        print(f"  Val:   {len(val_loader.dataset)} events ({len(val_loader)} batches)")
        print(f"  Test:  {len(test_loader.dataset)} events ({len(test_loader)} batches)")

    if config.verbose:
        print(f"\nCreating {config.model_type} model...")

    model = create_model(config)

    if config.verbose:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")

    start_epoch = 0
    if args.resume is not None:
        if config.verbose:
            print(f"\nResuming from checkpoint: {args.resume}")

        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

        if config.verbose:
            print(f"  Resuming from epoch {start_epoch}")

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

    if args.resume is not None and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = start_epoch

    if config.verbose:
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80 + "\n")

    results = trainer.train()

    if config.verbose:
        print("\n" + "="*80)
        print("Training Complete")
        print("="*80)
        print(f"\nEpochs trained: {results['epochs_trained']}")
        print(f"Time elapsed: {results['time_elapsed']/60:.2f} minutes")
        print("\nFinal metrics:")
        for key, val in results['final_metrics'].items():
            print(f"  {key}: {val:.6f}")

    save_results(results, output_path, prefix='training_results')

    if config.verbose:
        print(f"\nResults saved to {output_path}")
        print(f"Best model checkpoint: {output_path / 'best_model.pt'}")
        if config.use_tensorboard:
            print(f"TensorBoard logs: {output_path / 'logs'}")
            print(f"  View with: tensorboard --logdir {output_path / 'logs'}")

    return results


if __name__ == '__main__':
    main()
