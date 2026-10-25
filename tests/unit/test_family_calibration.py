"""Per-family recalibration, checked where the truth is known.

Synthetic families are built with a known offset: at the same model score, a
drive of family A is more likely to fail than the fleet, B less. The
calibrator should recover those offsets, shrink a family with little
evidence toward the fleet, leave the ranking within each family untouched,
and - the point of the whole exercise - help only when a family effect is
really there, and judged only out of sample.
"""

import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.agent.family_calibration import FamilyCalibrator, log_loss, select_tau
from src.agent.point_in_time import compare_scorings

SPEC = importlib.util.spec_from_file_location(
    "evaluate_family_calibration", Path("scripts/evaluate_family_calibration.py")
)
assert SPEC and SPEC.loader
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)

TRUE = {"A": 1.5, "B": -1.5, "C": 0.0}


def _fleet(
    dates,
    *,
    offsets=TRUE,
    drives_per_family=3000,
    small_family=None,
    seed=0,
) -> pl.DataFrame:
    """Each drive has a stable latent score; its label on each date is drawn
    from sigmoid(z + offset_family). The model only sees sigmoid(z)."""
    rng = np.random.default_rng(seed)
    families = dict(offsets)
    sizes = {f: drives_per_family for f in families}
    if small_family:
        name, offset, size = small_family
        families[name], sizes[name] = offset, size
    rows = []
    for family, offset in families.items():
        z = rng.normal(-4.0, 1.5, sizes[family])
        for day in dates:
            zt = z + rng.normal(0, 0.2, len(z))
            y = rng.random(len(z)) < 1 / (1 + np.exp(-(zt + offset)))
            for i in range(len(z)):
                rows.append(
                    (f"{family}{i}", day, float(1 / (1 + np.exp(-zt[i]))), int(y[i]), family)
                )
    return pl.DataFrame(
        rows, schema=["drive_id", "date", "p_fail", "label", "drive_model"], orient="row"
    ).with_columns(pl.col("label").cast(pl.Int8))


FIT_DATES = [dt.date(2026, 3, d) for d in (3, 10, 17, 24)]
TEST_DATES = [dt.date(2026, 4, d) for d in (5, 12)]


@pytest.fixture(scope="module")
def fit_obs():
    return _fleet(FIT_DATES, small_family=("S", 2.0, 60), seed=1)


def test_the_offsets_recover_the_true_family_effects(fit_obs):
    cal = FamilyCalibrator(tau=2.0).fit(fit_obs)
    # Offsets are relative to the fleet intercept, so compare differences.
    assert cal.offsets["A"] - cal.offsets["C"] == pytest.approx(1.5, abs=0.35)
    assert cal.offsets["B"] - cal.offsets["C"] == pytest.approx(-1.5, abs=0.45)


def test_a_family_with_little_evidence_is_pulled_toward_the_fleet(fit_obs):
    """S truly sits at +2.0 but has 60 drives. The prior pulls every offset
    toward ZERO - the fleet intercept - not toward any other family, so the
    shrinkage is measured against zero. (Measured against C it would mix in
    C's own offset: C is large enough to sit wherever its evidence puts it.)
    A tight prior must shrink S much harder than a loose one, while a large
    family barely moves."""
    tight = FamilyCalibrator(tau=0.1).fit(fit_obs)
    loose = FamilyCalibrator(tau=2.0).fit(fit_obs)
    assert abs(tight.offsets["S"]) < abs(loose.offsets["S"])
    assert abs(tight.offsets["S"]) < 0.5
    # A has ~3,000 drives: its offset hardly depends on the prior.
    assert tight.offsets["A"] == pytest.approx(loose.offsets["A"], abs=0.25)


def test_a_family_unseen_while_fitting_gets_the_fleet_calibration(fit_obs):
    cal = FamilyCalibrator(tau=1.0).fit(fit_obs)
    new = pl.DataFrame({"p_fail": [0.05], "drive_model": ["NEVER-SEEN"]})
    expected = 1 / (1 + np.exp(-(cal.a + cal.b * np.log(0.05 / 0.95))))
    assert cal.transform(new)[0] == pytest.approx(expected)


def test_the_ranking_inside_a_family_is_unchanged(fit_obs):
    """The map is monotone within a family, so only cross-family order moves."""
    cal = FamilyCalibrator(tau=1.0).fit(fit_obs)
    calibrated = cal.transform(fit_obs)
    for family in ("A", "B", "C"):
        mask = (fit_obs["drive_model"] == family).to_numpy()
        raw_order = np.argsort(fit_obs["p_fail"].to_numpy()[mask], kind="stable")
        cal_order = np.argsort(calibrated[mask], kind="stable")
        assert np.array_equal(raw_order, cal_order)


def test_tau_is_chosen_above_zero_when_families_differ(fit_obs):
    tau, scores = select_tau(fit_obs)
    assert tau > 0
    assert {r["tau"] for r in scores} >= {0.0, 0.5}


def test_tau_stays_small_when_there_is_no_family_effect():
    """With identical families the offsets can only fit noise, so the
    fleet-only baseline (tau=0) or a tight prior must win."""
    flat = _fleet(FIT_DATES, offsets={"A": 0.0, "B": 0.0, "C": 0.0}, seed=2)
    tau, _ = select_tau(flat)
    assert tau <= 0.25


def test_recalibration_improves_out_of_sample_fit_when_families_differ(fit_obs):
    test = _fleet(TEST_DATES, small_family=("S", 2.0, 60), seed=3)
    y = (test["label"].to_numpy() == 1).astype(float)
    fleet_only = FamilyCalibrator(tau=1e-6).fit(fit_obs)
    family = FamilyCalibrator(tau=1.0).fit(fit_obs)
    assert log_loss(y, family.transform(test)) < log_loss(y, fleet_only.transform(test))


# --- the evaluation, end to end ---------------------------------------------------


def test_the_evaluation_finds_a_real_gain_when_families_differ(fit_obs):
    test = _fleet(TEST_DATES, small_family=("S", 2.0, 60), seed=4)
    result = script.evaluate(fit_obs, test, n_boot=100)
    gains = [r for r in result["comparison"] if r["difference_ci"][0] > 0]
    assert gains, "a known family effect should give a recall gain clear of zero"
    assert result["tau"] > 0


def test_the_evaluation_finds_nothing_when_there_is_nothing_to_find():
    flat = {"A": 0.0, "B": 0.0, "C": 0.0}
    fit = _fleet(FIT_DATES, offsets=flat, seed=5)
    test = _fleet(TEST_DATES, offsets=flat, seed=6)
    result = script.evaluate(fit, test, n_boot=100)
    for row in result["comparison"]:
        lo, hi = row["difference_ci"]
        assert lo <= 0.02 and hi >= -0.02, f"spurious effect at {row['false_alarm_budget']}"


def test_fitting_on_or_after_the_test_window_is_refused(fit_obs):
    test = _fleet(TEST_DATES, seed=7)
    with pytest.raises(SystemExit, match="before the test window"):
        script.evaluate(test, fit_obs, n_boot=10)  # swapped
    overlapping = pl.concat([fit_obs, test.filter(pl.col("date") == TEST_DATES[0])])
    with pytest.raises(SystemExit, match="before the test window"):
        script.evaluate(overlapping, test, n_boot=10)


def test_a_missing_family_column_is_refused(fit_obs):
    test = _fleet(TEST_DATES, seed=8).drop("drive_model")
    with pytest.raises(SystemExit, match="drive_model"):
        script.evaluate(fit_obs, test, n_boot=10)


def test_identical_scorings_compare_to_exactly_zero():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = rng.random(500) < 0.1
    rows = compare_scorings(
        pl.Series([f"d{i}" for i in range(500)]), y, p, p.copy(), budgets=(0.05,), n_boot=20
    )
    assert rows[0]["difference"] == 0.0
    assert rows[0]["difference_ci"] == (0.0, 0.0)
