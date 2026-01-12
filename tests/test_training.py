# tests/test_artifacts.py
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


class TinyDataset(torch.utils.data.Dataset):
    """A minimal MNIST-like dataset: (1, 28, 28) images + labels 0-9."""
    def __init__(self, n: int = 32):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 1, 28, 28, generator=g)
        self.y = torch.randint(0, 10, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def test_train_creates_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Run everything in a temporary directory so we can assert on created files
    monkeypatch.chdir(tmp_path)

    # Import after chdir so your Path("models").mkdir(...) at import-time
    # creates folders inside tmp_path, not your repo.
    import my_project.train as train_module

    # Force CPU so the test is stable across machines
    monkeypatch.setattr(train_module, "DEVICE", torch.device("cpu"))

    # Replace corrupt_mnist() with a tiny deterministic dataset
    def fake_corrupt_mnist():
        ds = TinyDataset(n=32)
        return ds, ds  # train_set, test_set (second value unused in your train())

    monkeypatch.setattr(train_module, "corrupt_mnist", fake_corrupt_mnist)

    # Run a very small training
    train_module.train(lr=1e-3, batch_size=8, epochs=1)

    # Assert artifacts exist
    model_path = tmp_path / "models" / "model.pth"
    fig_path = tmp_path / "reports" / "figures" / "training_statistics.png"

    assert model_path.exists(), "Expected model checkpoint to be saved"
    assert fig_path.exists(), "Expected training statistics figure to be saved"

    # Optional: sanity check they’re non-empty files
    assert model_path.stat().st_size > 0
    assert fig_path.stat().st_size > 0

