# Running experiments

[Back to the project overview](../README.md)

## Environment

Use Python 3.10 or 3.11 and install the packages in [`requirements.txt`](../requirements.txt) inside an isolated environment.

For CUDA 12.4 on an NVIDIA system, install PyTorch from the official index before installing the remaining requirements:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

See PyTorch's [version-specific installation instructions](https://pytorch.org/get-started/previous-versions/) for CPU and other builds. Configure TensorFlow GPU support separately for CNN and MLP experiments. Feature extraction and model training can also use separate environments, exchanging features through CSV files.

## Script inputs

Run each experiment in a fresh Python process or notebook kernel. Prepare the following inputs for the selected model:

| Model family | Variables or configuration |
| --- | --- |
| `MLP-*` | `betti_train`, `betti_val`, `betti_test`, and matching `labels_*` arrays |
| `tabtransformer-*` | The same variables; multiclass requires pandas DataFrames, while the binary runner also accepts NumPy arrays |
| `CNN-*` | `image_train`, `image_val`, `image_test`, and matching `labels_*` arrays |
| `T-CNN-*` | Image arrays, Betti DataFrames, and matching label arrays |
| `swinv2-*` | Image and label arrays; set the device in the script |
| `t-swinv2-*` | Set the six NPZ/CSV paths in the `__main__` block, plus the class count and image configuration in `TSwinv2` |
| `BettiVectorization.py` | Call `generate_betti_vectors(images)` with an image array in BGR order |

For scripts using preloaded arrays, supply the variables through `runpy.run_path(..., run_name="__main__", init_globals=inputs)`, as shown in the README.

The [binary tabular transformer](../models/binaryclass/tabtransformer-binaryclass.py) also exposes `run_experiment`, `train_model`, and `evaluate` functions. To execute it with preloaded arrays:

```python
import runpy

# Use the inputs dictionary constructed in the README's data-loading example.
runpy.run_path(
    "models/binaryclass/tabtransformer-binaryclass.py",
    run_name="__main__",
    init_globals=inputs,
)
```

Importing this module makes its functions available without starting training.

## Feature conventions

The extractor uses cubical persistence with coefficient field 3 and homology dimensions 0 and 1. Each curve uses 50 bins. The 400 output columns are concatenated as:

```text
gray H0 | gray H1 | red H0 | red H1 | blue H0 | blue H1 | green H0 | green H1
```

Use OpenCV BGR channel order for extraction, including grayscale conversion through `cv2.COLOR_BGR2GRAY`. Convert RGB input arrays to BGR when loading images through other libraries.

Save feature CSVs with a header row and `index=False`, containing exactly 400 numeric columns. Match the row order of the corresponding image and label arrays. Keep identifiers and labels outside the feature matrix.

## Model configuration

- Set `num_classes` and image dimensions for the selected dataset. Several multiclass scripts default to three classes; the multiclass tabular transformer defaults to five.
- Set the epoch limit in the selected script. Several scripts use 300 epochs; the multiclass hybrid Swin entry point uses 100.
- The binary tabular transformer uses learning-rate multipliers `1.0`, `0.01`, and `0.001` across its three schedule stages. Several other models use `1.0`, `0.1`, and `0.01`.
- Set `pretrained=False` in the binary hybrid Swin constructor to initialize the vision backbone without pretrained weights. Its default is `True`.
- Use `checkpoint_path` in the binary training functions to select the checkpoint file. Other scripts write `best_model.pth` or `best_hybrid_model.pth` in the working directory; use a separate directory for each experiment.

## Evaluation

The binary training functions select checkpoints using validation loss and then evaluate the held-out test split. Each image batch is paired with its corresponding topological features, and each split has its own prediction accumulators. Losses are averaged over samples, including partial final batches.

Record dataset versions, split IDs, preprocessing, class mappings, random seeds, model settings, and the checkpoint-selection rule alongside each run. Retain prediction outputs with the reported metrics.
