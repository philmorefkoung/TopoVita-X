# Reproducibility notes

[Back to the project overview](../README.md)

These notes describe the code currently in this repository. The [paper](https://www.medrxiv.org/content/10.1101/2025.01.21.25320908v1) remains the source for the study protocol and reported results. Dataset splits and trained checkpoints are not included.

## Environment

[`requirements.txt`](../requirements.txt) targets Python 3.10/3.11 and retains the original package versions except for two dependency corrections and portable PyTorch version specifiers:

- `giotto-tda==0.6.2` requires `scikit-learn==1.3.2`; the previous `scikit-learn==1.5.2` pin conflicted with that requirement.
- scikit-learn 1.3.2 requires NumPy below 2.0; NumPy is now pinned to `1.26.4` rather than `2.0.2`. TensorFlow 2.18.0 also accepts this version.
- PyTorch and torchvision remain at `2.5.1` and `0.20.1`, with the CUDA-specific `+cu124` suffix removed so the platform's available distribution can be selected.

These constraints are recorded in the published [giotto-tda 0.6.2 metadata](https://pypi.org/pypi/giotto-tda/0.6.2/json), [scikit-learn 1.3.2 metadata](https://pypi.org/pypi/scikit-learn/1.3.2/json), and [TensorFlow 2.18.0 metadata](https://pypi.org/pypi/tensorflow/2.18.0/json). These changes resolve declared constraints; they do not establish numerical equivalence with the paper's environment.

For the original CUDA 12.4 PyTorch build on a supported NVIDIA system, install it from the official index before installing the remaining requirements:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

See PyTorch's [version-specific installation instructions](https://pytorch.org/get-started/previous-versions/) for CPU and other supported builds. TensorFlow has separate GPU platform requirements. The repository's PyTorch build selection does not configure TensorFlow GPU support.

Feature extraction and model training may also use separate environments because their interface is a CSV file. Record installed versions with `python -m pip freeze`. Dependency resolution alone does not validate every native library, GPU configuration, or training path.

## Script inputs

The model scripts are experiment templates rather than a packaged API or a unified command-line interface. Most execute training immediately when run. Run each experiment in a fresh Python process or notebook kernel because some scripts transform label arrays in place or reuse checkpoint filenames.

| Model family | Variables or configuration required before execution |
| --- | --- |
| `MLP-*` | `betti_train`, `betti_val`, `betti_test`, and matching `labels_*` arrays |
| `tabtransformer-*` | The same variables; Betti features must be pandas DataFrames because the scripts access `.values` |
| `CNN-*` | `image_train`, `image_val`, `image_test`, and matching `labels_*` arrays |
| `T-CNN-*` | Image arrays, Betti DataFrames, and matching label arrays |
| `swinv2-*` | Image and label arrays; inspect CUDA device selection in the script |
| `t-swinv2-*` | Edit the six NPZ/CSV paths in the `__main__` block; set class count and image configuration in `TSwinv2` |
| `BettiVectorization.py` | Call `generate_betti_vectors(images)` with an image array in BGR order |

For the scripts using preloaded arrays, the README's `runpy.run_path(..., init_globals=...)` pattern supplies the expected variables without changing their source. It does not validate a model's training or evaluation logic.

## Feature conventions

The extractor uses cubical persistence with coefficient field 3 and separate homology dimensions 0 and 1. Each curve uses 50 bins. The 400 output columns are concatenated as:

```text
gray H0 | gray H1 | red H0 | red H1 | blue H0 | blue H1 | green H0 | green H1
```

The source uses OpenCV BGR indexing and `cv2.COLOR_BGR2GRAY`. Match color order explicitly when loading images through other libraries. Keep model preprocessing and feature extraction consistent with the chosen experiment; do not silently interchange RGB and BGR.

CSV readers use pandas' default header handling. Save a header row and use `index=False`; an extra index, identifier, or label column would change the expected feature dimension. The code does not join features to images by an ID, so matching row order is essential.

## Configuration and evaluation details

Review these items before interpreting a new run as a reproduction:

- **Dataset preparation:** the repository does not provide split manifests, split-generation scripts, or a full image preprocessing pipeline. Record sample identities, label mappings, image resizing, channel order, scaling, and seeds in the experiment record.
- **Model defaults:** multiclass scripts often set `num_classes=3`. CNN and Swin image sizes are configured in source; changing only the dataset files does not adjust the model. Pretrained models may download weights on first use.
- **Epoch limits:** several scripts use 300 epochs, while the multiclass hybrid Swin entry point requests 100. Inspect each script's settings rather than assuming one shared configuration.
- **Learning-rate schedules:** the binary tabular transformer uses multipliers `1.0`, `0.01`, and `0.001`; several other scripts use `1.0`, `0.1`, and `0.01`. These are existing implementation differences.
- **Binary hybrid Swin:** in [`t-swinv2-binaryclass.py`](../models/binaryclass/t-swinv2-binaryclass.py), validation and test loops load `betti_features` but call the model with `features` left over from training. That can mispair images and features or cause a batch-size error. Correct and validate the evaluation path before using its metrics.
- **Binary tabular transformer:** [`tabtransformer-binaryclass.py`](../models/binaryclass/tabtransformer-binaryclass.py) appends test predictions to the lists from the last validation pass without clearing them. Its final metrics therefore combine validation and test samples. Reset these accumulators and validate evaluation before reporting test performance.
- **Checkpoint outputs:** multiple scripts write `best_model.pth` or `best_hybrid_model.pth` in the working directory. Use a separate run directory or preserve these files before another experiment overwrites them.

These observations document the released implementation; they do not establish whether the same issues affected the experiments reported in the paper. The documentation cleanup does not modify model or evaluation behavior.

## Recording a reproduction

Record the repository commit, dependency versions, hardware, dataset versions, split IDs, preprocessing choices, model configuration, seeds, and checkpoint-selection rule. Keep validation and test evaluation separate, and retain the raw prediction outputs used to calculate reported metrics.
