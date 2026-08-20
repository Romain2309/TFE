# Code Structure and Organization

This document explains the organization and architecture of the gravitational wave sky localization codebase.

---

## Directory Structure

```
submission/
├── gw_skylocalization/         # Main Python package
│   ├── __init__.py
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py          # Data loaders and datasets
│   │   ├── hdf5_loader.py      # HDF5 file reader
│   │   └── file_readers.py     # Various file format readers
│   ├── models/                 # Neural network architectures
│   │   ├── __init__.py
│   │   ├── backbones.py        # Feature extraction networks
│   │   ├── heads.py            # Task-specific output layers
│   │   ├── networks.py         # Complete model definitions
│   │   └── losses.py           # Loss functions
│   ├── training/               # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── config.py           # Configuration management
│   │   └── callbacks.py        # Training callbacks
│   ├── evaluation/             # Evaluation and metrics
│   │   ├── __init__.py
│   │   └── metrics.py          # Performance metrics
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── coordinates.py      # Coordinate transformations
│       ├── physics.py          # Physics calculations
│       └── io.py               # I/O utilities
├── scripts/                    # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_with_skymaps.py
│   ├── compare_models.py
│   ├── create_analysis_plots.py
│   └── preprocess_dataset.py
├── figures/                    # Generated visualizations
├── docs/                       # Documentation
├── README.md
├── requirements.txt
└── setup.py
```

---

## Module Descriptions

### `gw_skylocalization.data`

**Purpose**: Data loading and preprocessing

#### `loaders.py`
- `GWDataset`: PyTorch Dataset for gravitational wave data
- `create_dataloaders()`: Factory function for train/val/test loaders
- `collate_fn()`: Custom batch collation

**Key classes**:
```python
class GWDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, nside, transform=None):
        # Load data from HDF5
        # Apply preprocessing transforms

    def __getitem__(self, idx):
        # Return (strain_data, sky_position, healpix_pixel)
```

#### `hdf5_loader.py`
- `HDF5Loader`: Efficient loading of large HDF5 files
- Lazy loading for memory efficiency
- Caching for repeated access

#### `file_readers.py`
- Support for multiple file formats (HDF5, NumPy, etc.)
- Automatic format detection
- Validation and error checking

---

### `gw_skylocalization.models`

**Purpose**: Neural network architectures

#### `backbones.py`

Feature extraction networks:

```python
class Conv1DBackbone(nn.Module):
    """1D CNN for time-series feature extraction"""
    def __init__(self, in_channels=3, base_channels=16):
        # Convolutional layers
        # Batch normalization
        # Pooling layers

    def forward(self, x):
        # Input: (batch, 3, 2048)
        # Output: (batch, feature_dim)
```

**Architecture**:
- 4-6 convolutional layers
- Kernel sizes: 7-15
- Channel progression: 16 → 32 → 64 → 128 → 256
- Max pooling after each conv block
- Batch normalization
- ReLU activation

#### `heads.py`

Task-specific output layers:

```python
class RegressionHead(nn.Module):
    """Regression to 3D unit vector"""
    def forward(self, features):
        # Output: (batch, 3) - normalized unit vector

class ClassificationHead(nn.Module):
    """Classification over HEALPix pixels"""
    def forward(self, features):
        # Output: (batch, n_pixels) - logits

class ProbabilityMapHead(nn.Module):
    """Full-sky probability distribution"""
    def forward(self, features):
        # Output: (batch, n_pixels) - normalized probabilities
```

#### `networks.py`

Complete model definitions:

```python
class Regressor(nn.Module):
    def __init__(self, nside=16):
        self.backbone = Conv1DBackbone()
        self.head = RegressionHead()

class Classifier(nn.Module):
    def __init__(self, nside=16):
        self.backbone = Conv1DBackbone()
        self.head = ClassificationHead(n_pixels=12*nside**2)

class MultiTaskModel(nn.Module):
    def __init__(self, nside=16):
        self.backbone = Conv1DBackbone()
        self.regression_head = RegressionHead()
        self.classification_head = ClassificationHead(n_pixels=12*nside**2)

class ProbabilityMapModel(nn.Module):
    def __init__(self, nside=16):
        self.backbone = Conv1DBackbone()
        self.head = ProbabilityMapHead(n_pixels=12*nside**2)
```

#### `losses.py`

Loss functions:

```python
def regression_loss(predictions, targets):
    """MSE loss on unit sphere"""
    return F.mse_loss(predictions, targets)

def classification_loss(logits, pixel_indices):
    """Cross-entropy loss"""
    return F.cross_entropy(logits, pixel_indices)

def multitask_loss(reg_pred, cls_logits, target_vec, target_pixel,
                   reg_weight=1.0, cls_weight=1.0):
    """Combined regression + classification loss"""
    reg_loss = regression_loss(reg_pred, target_vec)
    cls_loss = classification_loss(cls_logits, target_pixel)
    return reg_weight * reg_loss + cls_weight * cls_loss

def probability_map_loss(prob_map, target_pixel, smoothness_weight=0.1):
    """Cross-entropy + spatial smoothness regularization"""
    ce_loss = F.cross_entropy(prob_map, target_pixel)
    smoothness = compute_spatial_smoothness(prob_map)
    return ce_loss + smoothness_weight * smoothness
```

---

### `gw_skylocalization.training`

**Purpose**: Training infrastructure

#### `trainer.py`

Main training loop:

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, criterion, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        for batch in self.train_loader:
            loss = self._train_step(batch)

    def validate(self):
        """Validate model"""
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self._validation_step(batch)

    def train(self, epochs, output_dir):
        """Full training loop with checkpointing"""
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.save_checkpoint(epoch, output_dir)
```

#### `config.py`

Configuration management:

```python
@dataclass
class TrainingConfig:
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = 'adam'

    # Training
    batch_size: int = 64
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 15

    # Model
    model_type: str = 'multitask'
    nside: int = 16
    dropout: float = 0.2

    # Data
    num_workers: int = 4
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
```

#### `callbacks.py`

Training callbacks:

```python
class EarlyStopping:
    """Stop training when validation loss stops improving"""

class LearningRateScheduler:
    """Adjust learning rate during training"""

class MetricsLogger:
    """Log training metrics to file"""

class CheckpointSaver:
    """Save model checkpoints"""
```

---

### `gw_skylocalization.evaluation`

**Purpose**: Model evaluation and metrics

#### `metrics.py`

Performance metrics:

```python
def calculate_angular_errors(predictions, targets):
    """
    Calculate angular errors between predicted and true positions.

    Args:
        predictions: (N, 3) - predicted unit vectors
        targets: (N, 3) - true unit vectors

    Returns:
        errors: (N,) - angular errors in degrees
    """
    cos_angle = (predictions * targets).sum(axis=1)
    cos_angle = np.clip(cos_angle, -1, 1)
    errors = np.arccos(cos_angle) * 180 / np.pi
    return errors

def calculate_searched_area(prob_map, true_pixel, credible_level=0.9):
    """
    Calculate searched area at given credible level.

    Args:
        prob_map: (N_pixels,) - probability distribution
        true_pixel: int - index of true source pixel
        credible_level: float - credible level (0-1)

    Returns:
        area: float - searched area in square degrees
    """
    # Sort pixels by probability (descending)
    sorted_pixels = np.argsort(prob_map)[::-1]

    # Find cumulative probability
    cumsum = np.cumsum(prob_map[sorted_pixels])

    # Find index where cumsum >= credible_level
    idx = np.searchsorted(cumsum, credible_level)

    # Calculate area
    pixel_area = healpy.nside2pixarea(nside, degrees=True)
    area = (idx + 1) * pixel_area

    return area

def evaluate_model(model, dataloader, device, nside):
    """
    Comprehensive model evaluation.

    Returns:
        metrics: dict with all performance statistics
    """
    all_errors = []
    all_areas = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            predictions = model(batch['strain_data'].to(device))
            errors = calculate_angular_errors(
                predictions.cpu().numpy(),
                batch['sky_positions'].numpy()
            )
            all_errors.extend(errors)

    return {
        'median_error': np.median(all_errors),
        'mean_error': np.mean(all_errors),
        'std_error': np.std(all_errors),
        'p90_error': np.percentile(all_errors, 90),
        'fraction_below_1deg': (np.array(all_errors) < 1.0).mean(),
        'fraction_below_5deg': (np.array(all_errors) < 5.0).mean(),
        'fraction_below_10deg': (np.array(all_errors) < 10.0).mean(),
    }
```

---

### `gw_skylocalization.utils`

**Purpose**: Utility functions

#### `coordinates.py`

Coordinate transformations:

```python
def unit_vector_to_radec(vec):
    """Convert 3D unit vector to RA, Dec"""
    x, y, z = vec
    ra = np.arctan2(y, x) * 180 / np.pi
    dec = np.arcsin(z) * 180 / np.pi
    return ra, dec

def radec_to_unit_vector(ra, dec):
    """Convert RA, Dec to 3D unit vector"""
    ra_rad = ra * np.pi / 180
    dec_rad = dec * np.pi / 180
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.array([x, y, z])

def healpix_pixel_to_unit_vector(pixel, nside):
    """Convert HEALPix pixel index to unit vector"""
    theta, phi = healpy.pix2ang(nside, pixel)
    return ang_to_vec(theta, phi)
```

#### `physics.py`

Physics calculations:

```python
def chirp_mass(m1, m2):
    """Calculate chirp mass"""
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

def merger_time(m1, m2, f_low):
    """Estimate time to merger"""
    M = chirp_mass(m1, m2)
    # Use Post-Newtonian approximation

def snr_in_detector(h, psd):
    """Calculate SNR in a detector"""
    return np.sqrt(4 * np.sum(np.abs(h)**2 / psd))
```

#### `io.py`

I/O utilities:

```python
def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']
```

---

## Scripts

### `train.py`

Main training script with argument parsing:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['regressor', 'classifier', 'multitask', 'probmap'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nside', type=int, default=16)
    # ... more arguments

    args = parser.parse_args()

    # Create model
    model = create_model(args.model, args.nside)

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(args.dataset, args.batch_size)

    # Train
    trainer = Trainer(model, train_loader, val_loader, ...)
    trainer.train(args.epochs, args.output)
```

### `evaluate.py`

Evaluation script:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    # ... more arguments

    args = parser.parse_args()

    # Load model
    model = torch.load(args.model_path)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, args.nside)

    # Save results
    save_metrics(metrics, args.output)
    create_plots(predictions, targets, args.output)
```

---

## Dependencies

### Core
- `torch`: Neural network framework
- `numpy`: Numerical computing
- `h5py`: HDF5 file I/O
- `healpy`: HEALPix operations

### Visualization
- `matplotlib`: Plotting
- `scipy`: Scientific computing

### Training
- `tqdm`: Progress bars
- `tensorboard`: Training visualization (optional)

---

For more information, see the README.md and USAGE_GUIDE.md files.
