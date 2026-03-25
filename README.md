# Deep Learning Detection of Aberrant Anterior Tibial Artery on Knee MRI

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0.1-orange.svg)](https://pytorch.org/)

Official code for the paper:

> **Automated Identification of Aberrant Anterior Tibial Artery on Knee MRI Using Deep Learning**
> Aihara et al., *Skeletal Radiology*, 2026
> [https://doi.org/10.1007/s00256-025-04912-3](https://doi.org/10.1007/s00256-025-04912-3)

---

## Overview

This repository provides the model architecture, training pipeline, and evaluation code used in the paper. We trained a ResNet10T-based binary classifier to detect the aberrant anterior tibial artery (AATA) — a surgically relevant vascular variant — from axial T2/PD-weighted knee MRI sequences.

The model operates at the MRI-slice level and aggregates predictions to the study level by averaging slice probabilities.

---

## Repository Structure

```
├── src/
│   ├── model.py      # KneeClassifier architecture and Grad-CAM
│   ├── dataset.py    # MRISliceDataset and transform utilities
│   └── metrics.py    # Evaluation metrics, bootstrap CI, ROC plotting
├── weights/
│   └── model.pth     # Pretrained model checkpoint
├── outputs/          # Generated figures and Grad-CAM PDFs (created at runtime)
├── 01_training.ipynb            # Training + internal test evaluation
├── 02_external_validation.ipynb # External test evaluation (UNIFESP)
└── requirements.txt
```

---

## Model

The classifier uses a ResNet10T backbone pretrained on ImageNet (via the [timm](https://github.com/huggingface/pytorch-image-models) library), with a lightweight head for binary classification:

```
ResNet10T features → Flatten → Dropout(0.3) → Linear(512→8) → ReLU → Dropout(0.3) → Linear(8→1)
```

- ~5.44 million trainable parameters
- Input: 224 × 224 × 3 (grayscale replicated across channels)
- Output: single logit (sigmoid → probability of AATA)

---

## Data

Training and internal evaluation used **2,315 knee MRI examinations** from Diagnóstico da América S.A. (DASA), Brazil. External validation used **617 examinations** from UNIFESP. Images were acquired on 1.5 T and 3.0 T scanners; only axial T2 and PD sequences without IV contrast were included.

**Images are not publicly available** due to patient privacy regulations. The preprocessing pipeline (DICOM → intensity normalization → JPEG) is described in the paper's Methods section.

| Split         | Studies | AATA-positive |
|---------------|--------:|:-------------:|
| Train (folds 2–4) | 1,389 | 37.7% |
| Validation (fold 1) | 463 | 37.7% |
| Internal test (fold 0) | 463 | 37.7% |
| External test (UNIFESP) | 617 | 6.0% |

---

## Pretrained Weights

The checkpoint at `weights/model.pth` corresponds to the model reported in the paper.
Load it with:

```python
import torch
from src.model import KneeClassifier

model = KneeClassifier(num_classes=1)
model.load_state_dict(torch.load("weights/model.pth", map_location="cpu"))
model.eval()
```

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training. Inference can run on CPU.

---

## Usage

Open and run the notebooks in order:

1. **`01_training.ipynb`** — Training and internal test evaluation.
   Set `IMAGE_FOLDER` and `METADATA_CSV` at the top of the configuration cell.

2. **`02_external_validation.ipynb`** — External test evaluation.
   Set `IMAGE_FOLDER` and `METADATA_CSV` for the external dataset.

Both notebooks import from `src/` and must be run from the repository root.

---

## Results

| Metric      | Internal Test | External Test |
|-------------|:-------------:|:-------------:|
| AUC         | 0.982         | 0.971         |
| F1-Score    | 0.978         | 0.786         |
| Sensitivity | 0.975         | 0.972         |
| Specificity | 0.989         | 0.969         |
| Accuracy    | 0.984         | 0.969         |

Study-level predictions using a threshold of **0.1717** (optimized on the validation set).

---

## Citation

```bibtex
@article{aihara2026aata,
  title   = {Automated Identification of Aberrant Anterior Tibial Artery on Knee {MRI} Using Deep Learning},
  author  = {Aihara, Tomás Ortiz and others},
  journal = {Skeletal Radiology},
  year    = {2026},
  doi     = {10.1007/s00256-025-04912-3}
}
```

---

## License

This code is released under the [MIT License](LICENSE).
