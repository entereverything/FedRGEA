# FedRGEA — Federated Robustness Framework with Genetic Evolutionary Aggregation

Federated learning framework for medical image classification.

## 📋 Table of Contents

- [Setup](#setup)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Attack Scenarios](#attack-scenarios)
- [Parameters](#parameters)

## 🔧 Setup

```bash
conda create -n fedrgea python=3.10
conda activate fedrgea
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard scikit-learn tqdm
```

## 📊 Datasets

| Dataset | Task | Classes |
|---------|------|---------|
| brainTumor | Brain tumor classification | 4 |
| isic2019 | Skin lesion classification | 8 |
| ham10000 | Dermatoscopic image classification | 7 |

Expected directory layout:

```
data/
├── brainTumor/
├── ISIC_2019/
└── HAM10000/
```

Non-IID partitioning via Dirichlet distribution (`--alpha`; lower = more heterogeneous).

## 🚀 Quick Start

### Training

```bash
python train.py \
  --dataset brainTumor \
  --exp my_experiment \
  --n_clients 10 \
  --rounds 100 \
  --alpha 1.0 \
  --local_ep 1 \
  --base_lr 3e-4 \
  --gpu 0 \
  --use-immune-detection \
  --detection-threshold 0.8 \
  --mad-scale 3.5 \
  --client-attack-types 0,0,0,2,2,2,2,0,0,0
```

### Testing

```bash
python test.py --dataset ham10000 --exp my_experiment --gpu 0
```

## 📁 Project Structure

```
FedRGEA/
├── train.py                          # Training entry point
├── test.py                           # Testing entry point
├── val.py                            # Validation metrics
├── .gitignore
├── README.md
│
├── dataset/
│   ├── dataset.py                    # Dataset definitions
│   ├── get_dataset.py                # Dataset loader
│   ├── randaugment.py                # RandAugment
│   └── sample_dirichlet.py           # Dirichlet partition
│
├── networks/
│   ├── networks.py                   # Model definitions
│   ├── all_models.py                 # Model registry
│   └── efficientnet.py               # EfficientNet
│
├── utils/
│   ├── FedAvg.py                     # FedAvg aggregation
│   ├── fedHybrid_three_line.py       # Three-line aggregation
│   ├── local_training.py             # Client local training
│   ├── losses.py                     # Loss functions
│   ├── utils.py                      # Utilities
│   ├── sample_dirichlet.py           # Data partitioning
│   └── immune/
│       ├── __init__.py
│       └── detector.py               # Immune detector
│
└── outputs/
    └── <exp>/
        ├── models/
        │   └── best_model.pth
        └── logs/
```

## ⚔️ Attack Scenarios

Configured via `--client-attack-types`:

| ID | Attack |
|----|--------|
| 0 | normal |
| 1 | constant gradient |
| 2 | sign-flip |
| 3 | random gradients |
| 4 | update-scaling |
| 5 | IPM |
| 6 | LIE |

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | brainTumor | Dataset name |
| `--n_clients` | 10 | Number of clients |
| `--rounds` | 100 | Communication rounds |
| `--alpha` | 1.0 | Dirichlet parameter (lower = more non-IID) |
| `--local_ep` | 1 | Local epochs |
| `--base_lr` | 3e-4 | Learning rate |
| `--detection-threshold` | 0.8 | Detection strictness [0,1] |
| `--mad-scale` | 3.5 | MAD threshold multiplier |
| `--csea-pop-size` | 8 | Population size |
| `--client-attack-types` | (all 0) | Attack type per client |
