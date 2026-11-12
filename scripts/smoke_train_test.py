"""One CPU optimizer step and held-out evaluation of the actual binary models.

Synthetic samples exercise execution, not diagnostic performance. No pretrained
weights or external datasets are downloaded. Checkpoints live in a temp directory.
"""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def load_model(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('-', '_'),
                                                 ROOT / 'models' / 'binaryclass' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_result(history, metrics, before, after):
    assert len(history['train_loss']) == 1
    assert all(np.isfinite(values).all() for values in history.values())
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics['n_samples'] == 3
    assert not torch.equal(before, after.detach().cpu()), 'No parameter update occurred.'
    return {'history': history, 'test_metrics': metrics, 'parameter_updated': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON result file.')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    splits = {}
    for name, count in [('train', 2), ('val', 2), ('test', 3)]:
        splits[name] = {
            'images': rng.integers(0, 256, (count, 128, 128, 3), dtype=np.uint8),
            'features': rng.normal(size=(count, 400)).astype(np.float32),
            'labels': np.arange(count, dtype=np.int64) % 2,
        }

    tabular = load_model('tabtransformer-binaryclass.py')
    hybrid = load_model('t-swinv2-binaryclass.py')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tab_loaders = []
        for name, data in splits.items():
            np.savez(tmp / f'{name}.npz', images=data['images'], labels=data['labels'])
            pd.DataFrame(data['features']).to_csv(tmp / f'{name}.csv', index=False)
            tab_loaders.append(DataLoader(TensorDataset(
                torch.from_numpy(data['features']), torch.from_numpy(data['labels']).float(),
            ), batch_size=2, shuffle=False))

        # Default 400-feature, 128-wide, five-layer TabTransformer architecture.
        tab_model = tabular.TabTransformer(num_features=400)
        before = tab_model.classifier.weight.detach().clone()
        tab_model, history, metrics = tabular.train_model(
            tab_model, *tab_loaders, num_epochs=1, device='cpu',
            checkpoint_path=tmp / 'tabular.pth',
        )
        tab_result = check_result(history, metrics, before, tab_model.classifier.weight)

        # Actual timm Swin-V2 Tiny backbone, default 128x128 input, and fusion network.
        # pretrained=False is essential: this check performs no model download.
        hybrid_model = hybrid.TSwinv2(pretrained=False)
        before = hybrid_model.fusion[-1].weight.detach().clone()
        vision_before = hybrid_model.vision_model.patch_embed.proj.weight.detach().clone()
        topology_before = hybrid_model.mlp[0].weight.detach().clone()
        hybrid_loaders = hybrid.load_presplit_data(
            *(tmp / f'{name}.npz' for name in splits),
            *(tmp / f'{name}.csv' for name in splits), batch_size=2,
        )
        hybrid_model, history, metrics = hybrid.train_model(
            hybrid_model, *hybrid_loaders, num_epochs=1, device='cpu',
            checkpoint_path=tmp / 'hybrid.pth',
        )
        hybrid_result = check_result(history, metrics, before, hybrid_model.fusion[-1].weight)
        assert not torch.equal(vision_before, hybrid_model.vision_model.patch_embed.proj.weight.detach())
        assert not torch.equal(topology_before, hybrid_model.mlp[0].weight.detach())
        hybrid_result.update(vision_branch_updated=True, topology_branch_updated=True)

    results = {
        'seed': 42,
        'device': 'cpu',
        'torch_threads': 2,
        'data': 'independently generated synthetic samples; not a paper reproduction',
        'split_sizes': {name: len(data['labels']) for name, data in splits.items()},
        'epochs': 1,
        'optimizer_steps_per_model': 1,
        'pretrained': False,
        'versions': {name: importlib.metadata.version(name) for name in
                     ['torch', 'torchvision', 'timm', 'numpy', 'pandas', 'scikit-learn']},
        'tabular_transformer': tab_result,
        'hybrid_swin': hybrid_result,
    }
    serialized = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
