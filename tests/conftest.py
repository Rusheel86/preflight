import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def clean_loader():
    x = torch.randn(40, 3, 32, 32)
    y = torch.randint(0, 5, (40,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def nan_loader():
    x = torch.randn(40, 3, 32, 32)
    x[5, 0, 0, 0] = float("nan")
    x[10, 1, 2, 3] = float("inf")
    y = torch.randint(0, 5, (40,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def nhwc_loader():
    x = torch.randn(40, 32, 32, 3)  # NHWC
    y = torch.randint(0, 5, (40,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def imbalanced_loader():
    x = torch.randn(100, 3, 32, 32)
    # class 0: 95 samples, class 1: 5 samples
    y = torch.cat([torch.zeros(95, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    return DataLoader(TensorDataset(x, y), batch_size=10)


@pytest.fixture
def leaky_loaders():
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 3, (20,))
    full_ds = TensorDataset(x, y)
    # both loaders share the same data → 100% leak
    train_loader = DataLoader(full_ds, batch_size=5)
    val_loader = DataLoader(full_ds, batch_size=5)
    return train_loader, val_loader
