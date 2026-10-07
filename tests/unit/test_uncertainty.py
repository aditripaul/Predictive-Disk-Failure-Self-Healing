import numpy as np

from src.models.access_log import count_test_accesses, record_test_access
from src.models.uncertainty import drive_level_bootstrap_ci, paired_difference_ci


def _drives(seed: int, n_drives: int = 400, days: int = 5):
    """Failing drives score high, healthy drives score low, with some overlap."""
    rng = np.random.default_rng(seed)
    failing = rng.random(n_drives) < 0.2
    drive_ids = np.repeat([f"d{i}" for i in range(n_drives)], days)
    y = np.repeat(failing, days).astype(int)
    base = np.repeat(np.where(failing, 0.7, 0.3) + rng.normal(0, 0.15, n_drives), days)
    scores = np.clip(base + rng.normal(0, 0.02, len(y)), 0, 1)
    return drive_ids, y, scores


def test_ci_brackets_point_estimate():
    ids, y, s = _drives(0)
    out = drive_level_bootstrap_ci(ids, y, s, threshold=0.5, n_boot=300)
    lo, hi = out["precision_ci"]
    assert lo <= out["precision"] <= hi
    lo, hi = out["recall_ci"]
    assert lo <= out["recall"] <= hi
    assert hi - lo > 0


def test_ci_is_deterministic_for_a_seed():
    ids, y, s = _drives(1)
    a = drive_level_bootstrap_ci(ids, y, s, threshold=0.5, n_boot=100, seed=7)
    b = drive_level_bootstrap_ci(ids, y, s, threshold=0.5, n_boot=100, seed=7)
    assert a == b


def test_identical_variants_have_zero_difference():
    ids, y, s = _drives(2)
    out = paired_difference_ci(ids, y, s, s, 0.5, 0.5, n_boot=100)
    assert out["difference"] == 0.0
    assert out["difference_ci"] == [0.0, 0.0]


def test_access_log_appends_and_counts(tmp_path):
    log = tmp_path / "log.jsonl"
    assert count_test_accesses(log) == 0
    record_test_access("variant_a", "validation of threshold", log_path=log)
    record_test_access("variant_b", "headline result", log_path=log)
    assert count_test_accesses(log) == 2
    lines = log.read_text().splitlines()
    assert '"variant": "variant_a"' in lines[0]
