from preflight.checks.resources import check_class_imbalance, check_vram
from preflight.registry import Severity


def test_vram_skipped_no_model(clean_loader):
    r = check_vram(dataloader=clean_loader)
    assert r.passed


def test_class_imbalance_passes(clean_loader):
    r = check_class_imbalance(dataloader=clean_loader)
    assert r.passed


def test_class_imbalance_detects_imbalance(imbalanced_loader):
    r = check_class_imbalance(dataloader=imbalanced_loader)
    assert not r.passed
    assert r.severity == Severity.WARN
    assert r.fix_hint is not None
