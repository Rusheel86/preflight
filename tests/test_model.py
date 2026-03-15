import pytest
import torch.nn as nn

from preflight.checks.model import check_gradients, check_shape_mismatch
from preflight.registry import Severity


@pytest.fixture
def simple_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))


def test_shape_mismatch_skipped_no_model(clean_loader):
    r = check_shape_mismatch(dataloader=clean_loader)
    assert r.passed


def test_gradients_skipped_no_model(clean_loader):
    r = check_gradients(dataloader=clean_loader)
    assert r.passed


def test_gradients_passes_with_model(clean_loader):
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 5))
    loss_fn = nn.CrossEntropyLoss()
    r = check_gradients(dataloader=clean_loader, model=model, loss_fn=loss_fn)
    assert r.passed
    assert r.severity == Severity.FATAL
