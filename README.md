# TopoVita-X

Topological features and deep learning for blood-cell image classification.

[Paper](https://link.springer.com/chapter/10.1007/978-3-032-09513-8_44) · [Data](#data) · [Usage](#usage) · [Running guide](docs/REPRODUCIBILITY.md) · [Citation](#citation)

Research code for **Diagnosis of Blood Diseases and Disorders with Topological Deep Learning**, by Philmore Koung, Saba Fatema, Nagehan Pakasticali, Hung Luu, and Baris Coskunuzer. Published in *Machine Learning in Medical Imaging (MLMI 2025)*, Lecture Notes in Computer Science, volume 16241, pages 452–466, Springer, 2026.

## Overview

TopoVita-X combines persistent-homology descriptors with image-based models for classifying acute lymphoblastic leukemia, acute myeloid leukemia, malaria, and babesiosis images. The repository includes:

- **Topology-only models:** an MLP and a tabular transformer using 400-dimensional Betti vectors.
- **Image baselines:** CNNs and Swin Transformer V2.
- **Hybrid models:** CNN and Swin V2 image representations fused with topological features.

The code is organized as experiment scripts. Most expect data arrays loaded into their Python namespace; the hybrid Swin scripts contain editable file paths. See the [running guide](docs/REPRODUCIBILITY.md) for model configuration and script inputs.

## Repository map

| Path | Contents |
| --- | --- |
| [`betti_vectorization/BettiVectorization.py`](betti_vectorization/BettiVectorization.py) | Cubical persistence and Betti-curve extraction |
| [`models/binaryclass/`](models/binaryclass/) | Binary classification experiments |
| [`models/multiclass/`](models/multiclass/) | Multiclass classification experiments |
| [`plots/`](plots/) | Accuracy and AUC comparison figures |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Environment setup, script inputs, and model configuration |
| [`CITATION.cff`](CITATION.cff) | Machine-readable paper citation |

Each model directory contains `MLP-*`, `tabtransformer-*`, `CNN-*`, `T-CNN-*`, `swinv2-*`, and `t-swinv2-*` scripts. The `T-` / `t-` prefixes identify hybrid image-and-topology models.

## Setup

Use Python 3.10 or 3.11 in an isolated environment:

```bash
git clone https://github.com/philmorefkoung/TopoVita-X.git
cd TopoVita-X
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The scripts use TensorFlow/Keras for CNN and MLP models, PyTorch/timm for transformer models, and giotto-tda/OpenCV for feature extraction. For NVIDIA CUDA 12.4, see the [GPU installation notes](docs/REPRODUCIBILITY.md#environment). Set the selected script's device to match your CPU or GPU.

## Data

Obtain data from the original providers below and prepare train, validation, and test splits.

| Dataset | Source |
| --- | --- |
| ALL-IDB | [University of Milan ALL-IDB](https://scotti.di.unimi.it/all/) |
| AML | [TCIA collection linked by the paper](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/) |
| Malaria | [NLM malaria datasets](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets) |
| Babesiosis / NIAID | [Dataset resource](https://data.niaid.nih.gov/resources?id=mendeley_38jtn4nzs6) |

For each split, prepare an `.npz` archive with `images` and `labels` arrays, plus a CSV of topological features:

```text
data/
  train.npz        # images: (N, H, W, 3); labels: (N,)
  val.npz
  test.npz
  train-betti.csv  # N rows, 400 numeric feature columns, header, no index
  val-betti.csv
  test-betti.csv
```

Keep image, label, and CSV row order identical. Labels must use the selected script's class encoding: `0`/`1` for binary tasks or consecutive integer class IDs for multiclass tasks. Match image dimensions and `num_classes` to the model configuration. The original experiment settings use 128 × 128 images for malaria and babesiosis and 224 × 224 for ALL and AML; several scripts default to 128 × 128.

## Usage

Run these examples from the repository root after preparing your environment and data.

### Extract Betti vectors

The extractor expects **OpenCV BGR channel order**. It concatenates 50-bin Betti-0 and Betti-1 curves for grayscale, red, blue, and green, in that order, producing 400 features per image. Convert RGB arrays to BGR before extraction when necessary.

```python
import numpy as np
from betti_vectorization.BettiVectorization import generate_betti_vectors

with np.load("data/train.npz") as data:
    images = data["images"]  # Must be BGR for this extractor.

betti = generate_betti_vectors(images)
betti.to_csv("data/train-betti.csv", index=False)
```

Repeat for validation and test, preserving sample order and preprocessing.

### Train a topology-only binary MLP

Load the variables expected by the existing MLP experiment:

```python
from pathlib import Path
import runpy
import numpy as np
import pandas as pd

data_dir = Path("data")
inputs = {}
for split in ("train", "val", "test"):
    with np.load(data_dir / f"{split}.npz") as data:
        labels = data["labels"].reshape(-1)
    betti = pd.read_csv(data_dir / f"{split}-betti.csv")
    assert betti.shape == (len(labels), 400)
    assert np.isfinite(betti.to_numpy(dtype=float)).all()
    assert set(np.unique(labels)).issubset({0, 1})
    inputs[f"labels_{split}"] = labels
    inputs[f"betti_{split}"] = betti

runpy.run_path("models/binaryclass/MLP-binaryclass.py", run_name="__main__", init_globals=inputs)
```

For other model families, consult the [script input table](docs/REPRODUCIBILITY.md#script-inputs). Configure class counts, image size, pretrained weights, and file paths before starting an experiment.

## Results

The figures below compare model accuracy and AUC. See the [paper](https://link.springer.com/chapter/10.1007/978-3-032-09513-8_44) for experimental protocols and detailed results.

### Accuracy

![Accuracy comparison of models across the study datasets](plots/RareBloodDiseaseAccuracyPlot.png)

### AUC

![AUC comparison of models across the study datasets](plots/RareBloodDiseaseAUCPlot.png)

## Citation

If you use this work, please cite the paper. GitHub's **Cite this repository** action is supported by [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{koung2026diagnosis,
  title     = {Diagnosis of Blood Diseases and Disorders with Topological Deep Learning},
  author    = {Koung, Philmore and Fatema, Saba and Pakasticali, Nagehan and Luu, Hung and Coskunuzer, Baris},
  booktitle = {Machine Learning in Medical Imaging},
  series    = {Lecture Notes in Computer Science},
  volume    = {16241},
  pages     = {452--466},
  year      = {2026},
  publisher = {Springer},
  address   = {Cham},
  doi       = {10.1007/978-3-032-09513-8_44},
  url       = {https://link.springer.com/chapter/10.1007/978-3-032-09513-8_44}
}
```

## Acknowledgements

We thank the dataset creators and the developers of the scientific Python libraries used in this work. Please also cite the datasets and methods relevant to your experiments.
