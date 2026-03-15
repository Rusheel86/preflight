from preflight.checks.splits import check_label_leakage, check_split_sizes
from preflight.registry import Severity


def test_leakage_detected(leaky_loaders):
    train_loader, val_loader = leaky_loaders
    r = check_label_leakage(
        dataloader=train_loader,
        config={"val_dataloader": val_loader},
    )
    assert not r.passed
    assert r.severity == Severity.FATAL


def test_leakage_skipped_no_val(clean_loader):
    r = check_label_leakage(dataloader=clean_loader)
    assert r.passed  # skipped, not failed


def test_split_sizes_passes(clean_loader):
    r = check_split_sizes(dataloader=clean_loader)
    assert r.severity == Severity.INFO


def test_split_sizes_no_dataset_attr(clean_loader):
    # should handle gracefully
    r = check_split_sizes(dataloader=clean_loader)
    assert r is not None
