"""Regression checks for split isolation and image/topology alignment."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def load_model(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('-', '_'),
                                                 ROOT / 'models' / 'binaryclass' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tabular = load_model('tabtransformer-binaryclass.py')
        cls.hybrid = load_model('t-swinv2-binaryclass.py')

    def test_tabular_metrics_contain_only_the_requested_split(self):
        class FixedLogits(nn.Module):
            def forward(self, features):
                return features[:, :1]

        def loader(logits, labels):
            return DataLoader(TensorDataset(
                torch.tensor(logits, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(labels, dtype=torch.float32),
            ), batch_size=2)

        model = FixedLogits()
        validation = loader([-5, 5, -4, 4], [1, 0, 1, 0])
        test = loader([-3, 3, -2], [0, 1, 0])
        val_metrics = self.tabular.evaluate(model, validation, 'cpu')
        test_metrics = self.tabular.evaluate(model, test, 'cpu')

        self.assertEqual(val_metrics['n_samples'], 4)
        self.assertEqual(val_metrics['accuracy'], 0)
        self.assertEqual(test_metrics['n_samples'], 3)
        for metric in ('accuracy', 'precision', 'recall', 'auc'):
            self.assertEqual(test_metrics[metric], 1)
        expected_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([-3.0, 3.0, -2.0]), torch.tensor([0.0, 1.0, 0.0]),
        ).item()
        self.assertAlmostEqual(test_metrics['loss'], expected_loss, places=7)
        self.assertEqual(test_metrics, self.tabular.evaluate(model, test, 'cpu'))

    def test_hybrid_uses_each_splits_features_and_handles_singleton_test_batch(self):
        class AlignedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.seen = []

            def forward(self, images, features):
                # Each image encodes its own feature ID; stale training features fail.
                torch.testing.assert_close(images[:, 0, 0, 0], features[:, 0])
                self.seen.extend(images[:, 0, 0, 0].detach().tolist())
                return features[:, :1] * self.scale

        def loader(markers):
            values = torch.tensor(markers, dtype=torch.float32)
            return DataLoader(TensorDataset(
                values.reshape(-1, 1, 1, 1), values.reshape(-1, 1),
                (values > 0).float(),
            ), batch_size=2)

        model = AlignedModel()
        with tempfile.TemporaryDirectory() as tmp:
            _, history, metrics = self.hybrid.train_model(
                model, loader([-5, 5]), loader([-3, 3, 4]), loader([-2, 2, -1]),
                num_epochs=1, device='cpu', checkpoint_path=Path(tmp) / 'best.pth',
            )

        self.assertEqual(model.seen, [-5, 5, -3, 3, 4, -2, 2, -1])
        self.assertEqual(metrics['n_samples'], 3)
        self.assertEqual(metrics['auc'], 1)
        self.assertTrue(np.isfinite(history['train_loss'][0]))
        self.assertTrue(np.isfinite(history['val_loss'][0]))
        expected_val_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([-3.0, 3.0, 4.0]) * model.scale.detach(),
            torch.tensor([0.0, 1.0, 1.0]),
        ).item()
        self.assertAlmostEqual(history['val_loss'][0], expected_val_loss, places=7)


if __name__ == '__main__':
    unittest.main()
