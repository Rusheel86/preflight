from preflight.checks.data import check_channel_ordering, check_nan_inf, check_normalisation
from preflight.registry import Severity


def test_nan_inf_passes_clean(clean_loader):
    r = check_nan_inf(dataloader=clean_loader)
    assert r.passed
    assert r.severity == Severity.FATAL


def test_nan_inf_fails_nan(nan_loader):
    r = check_nan_inf(dataloader=nan_loader)
    assert not r.passed
    assert r.severity == Severity.FATAL
    assert r.fix_hint is not None


def test_normalisation_passes_clean(clean_loader):
    r = check_normalisation(dataloader=clean_loader)
    assert r.passed


def test_channel_ordering_passes_nchw(clean_loader):
    r = check_channel_ordering(dataloader=clean_loader)
    assert r.passed


def test_channel_ordering_fails_nhwc(nhwc_loader):
    r = check_channel_ordering(dataloader=nhwc_loader)
    assert not r.passed
    assert "NHWC" in r.message
    assert r.fix_hint is not None
