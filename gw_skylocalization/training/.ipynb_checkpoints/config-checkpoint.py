

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List
from pathlib import Path
import json

@dataclass
class TrainingConfig:

    model_type: str = 'regressor'
    use_physics_features: bool = True
    strain_feature_dim: int = 256
    physics_feature_dim: int = 32

    data_dir: str = None
    target_length: int = 2048
    bandpass_low: float = 30.0
    bandpass_high: float = 1024.0
    normalize: bool = True

    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    max_events: Optional[int] = None

    healpix_nside: int = 8

    batch_size: int = 32
    n_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    loss_type: str = 'angular'
    tau_weight: float = 0.1
    label_smoothing: float = 0.0

    optimizer: str = 'adamw'
    momentum: float = 0.9

    scheduler_type: str = 'plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 10

    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.0

    save_best_only: bool = True
    save_last: bool = True
    monitor_metric: str = 'val_loss'

    num_workers: int = 4
    pin_memory: bool = True
    cache_train: bool = False

    seed: int = 42
    deterministic: bool = False

    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_frequency: int = 5

    use_tensorboard: bool = True
    log_interval: int = 10
    verbose: bool = True

    device: str = 'auto'
    mixed_precision: bool = False

    def __post_init__(self):
        
        valid_models = ['classifier', 'regressor', 'multitask', 'probmap']
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}, got {self.model_type}")

        if not (0 < self.train_frac < 1 and 0 < self.val_frac < 1 and 0 < self.test_frac < 1):
            raise ValueError("All split fractions must be between 0 and 1")
        if abs(self.train_frac + self.val_frac + self.test_frac - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {self.train_frac + self.val_frac + self.test_frac}")

        if self.bandpass_low >= self.bandpass_high:
            raise ValueError(f"bandpass_low ({self.bandpass_low}) must be < bandpass_high ({self.bandpass_high})")

        if self.experiment_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.model_type}_{timestamp}"

        self.data_dir = str(Path(self.data_dir))
        self.output_dir = str(Path(self.output_dir) / self.experiment_name)

    @property
    def bandpass(self) -> Tuple[float, float]:
        
        return (self.bandpass_low, self.bandpass_high)

    @property
    def n_pixels(self) -> int:
        
        return 12 * self.healpix_nside ** 2

    def to_dict(self) -> dict:
        
        return asdict(self)

    def save(self, path: str):
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def __str__(self) -> str:
        
        lines = ["Training Configuration:"]
        lines.append("=" * 60)

        categories = {
            "Model": ["model_type", "use_physics_features", "strain_feature_dim", "physics_feature_dim"],
            "Data": ["data_dir", "target_length", "bandpass_low", "bandpass_high", "normalize"],
            "Splits": ["train_frac", "val_frac", "test_frac", "max_events"],
            "Training": ["batch_size", "n_epochs", "learning_rate", "weight_decay", "gradient_clip"],
            "Optimization": ["optimizer", "scheduler_type", "early_stopping", "patience"],
            "Output": ["output_dir", "experiment_name"],
        }

        for category, keys in categories.items():
            lines.append(f"\n{category}:")
            for key in keys:
                value = getattr(self, key)
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

if __name__ == "__main__":

    config = TrainingConfig()

    custom_config = TrainingConfig(
        model_type='probmap',
        n_epochs=100,
        learning_rate=5e-4,
        batch_size=64,
        experiment_name='test_probmap'
    )

    custom_config.save('/tmp/test_config.json')
    loaded_config = TrainingConfig.load('/tmp/test_config.json')

    assert custom_config.experiment_name == loaded_config.experiment_name

