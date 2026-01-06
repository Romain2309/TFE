# Gravitational Wave Sky Localization

Deep learning models for gravitational wave source localization from binary black hole merger signals.

## Models

This project implements four architectures:
- **Regressor**: Direct unit vector regression
- **Classifier**: HEALPix pixel classification
- **Multi-Task**: Combined regression and classification
- **Probability Map**: Full-sky probability distribution

## Requirements

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy h5py healpy matplotlib pandas
```

Install package:

```bash
pip install -e .
```

## Repository Structure

```
submission/
├── data_generation/        # Dataset generation scripts
├── gw_skylocalization/     # Core package
│   ├── data/               # Data loading
│   ├── models/             # Neural networks
│   ├── training/           # Training infrastructure
│   ├── evaluation/         # Metrics
│   └── utils/              # Utilities
├── scripts/                # Training and evaluation scripts
└── results/                # Trained models (36 configurations)
```
