"""Split separation (src/labels/split_checks.py, src/labels/splits.py; dataset_strategy.md 9.5).

The chronological-with-gaps design of runbook 1.1.8 is the fixture; the drive-holdout design in
use (`OLD_DESIGN`) must keep producing the same split periods as before the embargo and
feature-statistics settings existed.
"""

from __future__ import annotations

import copy
import datetime as dt

import polars as pl
import pytest
import yaml

from src.labels.split_checks import (
    SplitSeparationError,
    check_label_table_separation,
    check_split_config,
    raise_on_violations,
    strict_separation_enabled,
)
from src.labels.splits import (
    add_chronological_split,
    apply_drive_level_holdout,
    embargo_days,
    feature_stats_cutoff,
    first_date_after_validation,
    first_test_date,
    resolved_test_end,
    sample_holdout_drive_ids,
    split_period_predicate,
)

D = dt.date

NEW_DESIGN = {
    "primary_horizon_days": 30,
    "splits": {
        "strict_separation": True,
        "train_start": "2026-01-01",
        "train_end": "2026-05-20",
        "validation_end": "2026-06-16",
        "test_embargo_days": 30,
        "test_end": "2026-08-31",
        "observation_end": "2026-09-30",
        "test_split": "sealed",
        "sealed_start": "2026-07-01",
        "feature_stats_end": "2026-04-21",
        "holdout_fraction": 0,
        "purge_label_window": True,
        "validation_after_train_only": True,
    },
}

# The drive-holdout design of runbook 1.1.8, as configs/model.yaml has it.
OLD_DESIGN = {
    "primary_horizon_days": 30,
    "splits": {
        "train_start": "2026-01-01",
        "train_end": "2026-05-01",
        "validation_end": "2026-05-01",
        "test_end": "2026-08-31",
        "observation_end": "2026-09-30",
        "test_split": "sealed",
        "sealed_start": "2026-07-01",
        "holdout_fraction": 0.2,
        "purge_label_window": False,
        "validation_after_train_only": False,
    },
}


def _with(config: dict, **splits) -> dict:
    out = copy.deepcopy(config)
    out["splits"].update(splits)
    return out


def _labels(config: dict) -> pl.LazyFrame:
    """One drive, a labelled row a day from 01-01 to 09-30, split as build-labels splits it."""
    days = pl.date_range(D(2026, 1, 1), D(2026, 9, 30), "1d", eager=True)
    frame = pl.DataFrame(
        {
            "drive_id": ["d"] * days.len(),
            "date": days,
            "horizon_days": [30] * days.len(),
            "label": [0] * days.len(),
        }
    )
    splits = config["splits"]
    return add_chronological_split(
        frame,
        train_end=splits["train_end"],
        validation_end=splits["validation_end"],
        sealed_start=splits.get("sealed_start"),
    ).lazy()


def _rules(violations) -> set[str]:
    return {v.rule for v in violations}


def test_new_design_passes_the_config_checks():
    assert check_split_config(NEW_DESIGN, 30) == []
    assert strict_separation_enabled(NEW_DESIGN)


def test_old_design_is_not_strict_and_the_checks_say_why():
    assert not strict_separation_enabled(OLD_DESIGN)
    rules = _rules(check_split_config(OLD_DESIGN, 30))
    assert {
        "purge_label_window",
        "validation_after_train_only",
        "holdout_fraction",
        "test_embargo_days",
        "feature_stats_end",
    } <= rules


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        ({"purge_label_window": False}, "purge_label_window"),
        ({"validation_after_train_only": False}, "validation_after_train_only"),
        ({"holdout_fraction": 0.2}, "holdout_fraction"),
        ({"test_embargo_days": 29}, "test_embargo_days"),
        ({"test_embargo_days": 0}, "test_embargo_days"),
        ({"feature_stats_end": None}, "feature_stats_end"),
        ({"feature_stats_end": "2026-05-21"}, "feature_stats_end"),
        ({"validation_end": "2026-05-20"}, "boundaries"),
        ({"test_end": "2026-07-10", "observation_end": None}, "test_end"),
    ],
)
def test_each_broken_rule_is_reported(change, rule):
    assert rule in _rules(check_split_config(_with(NEW_DESIGN, **change), 30))


def test_the_embargo_and_statistics_cutoff_must_cover_the_horizon_of_the_run():
    config = _with(NEW_DESIGN, test_embargo_days=44, feature_stats_end="2026-04-07")
    assert check_split_config(config, 44) == []
    # the same settings under a 60-day horizon: a shorter embargo, and a later last train row
    assert {"test_embargo_days", "feature_stats_end"} <= _rules(check_split_config(config, 60))


def test_feature_statistics_may_stop_exactly_after_the_last_train_row():
    # train_end 05-20 minus the 30-day horizon: the last train row is 04-20, so rows before
    # 04-21 are train rows only; 04-22 would include a row nothing was trained on.
    assert check_split_config(_with(NEW_DESIGN, feature_stats_end="2026-04-21"), 30) == []
    assert "feature_stats_end" in _rules(
        check_split_config(_with(NEW_DESIGN, feature_stats_end="2026-04-22"), 30)
    )


def test_label_table_of_the_new_design_passes():
    assert (
        check_label_table_separation(_labels(NEW_DESIGN), NEW_DESIGN, 30, test_split="sealed") == []
    )


def test_label_table_catches_a_test_period_inside_the_validation_labels():
    config = _with(NEW_DESIGN, test_embargo_days=0)
    found = check_label_table_separation(_labels(config), config, 30, test_split="sealed")
    assert any(v.rule == "label_horizon" and "test starts" in v.detail for v in found)


def test_label_table_catches_train_labels_running_into_validation():
    config = _with(NEW_DESIGN, purge_label_window=False)
    found = check_label_table_separation(_labels(config), config, 30, test_split="sealed")
    assert any(v.rule == "label_horizon" and "validation starts" in v.detail for v in found)


def test_label_table_reports_an_empty_role():
    config = _with(NEW_DESIGN, test_end="2026-07-01")  # the embargo leaves no test row
    found = check_label_table_separation(_labels(config), config, 30, test_split="sealed")
    assert any(v.rule == "empty" and "test" in v.detail for v in found)


def test_raise_on_violations_lists_every_failed_rule():
    violations = check_split_config(
        _with(NEW_DESIGN, purge_label_window=False, test_embargo_days=0), 30
    )
    with pytest.raises(SplitSeparationError) as caught:
        raise_on_violations(violations, where="the 30-day run")
    assert "purge_label_window" in str(caught.value)
    assert "test_embargo_days" in str(caught.value)
    raise_on_violations([], where="x")  # nothing to report: no error


def test_the_embargo_moves_the_first_test_date():
    frame = pl.DataFrame({"date": pl.date_range(D(2026, 6, 1), D(2026, 8, 31), "1d", eager=True)})
    predicate = split_period_predicate("test", NEW_DESIGN, 30)
    assert predicate is not None
    assert frame.filter(predicate)["date"].min() == D(2026, 7, 17)
    assert frame.filter(predicate)["date"].max() == D(2026, 8, 31)


def test_without_an_embargo_the_test_predicate_is_what_it_was():
    """The old design: test is capped at the last labelled day and has no lower bound."""
    assert embargo_days(OLD_DESIGN) == 0
    frame = pl.DataFrame({"date": pl.date_range(D(2026, 6, 1), D(2026, 9, 30), "1d", eager=True)})
    kept = frame.filter(split_period_predicate("test", OLD_DESIGN, 30))["date"]
    assert (kept.min(), kept.max()) == (D(2026, 6, 1), D(2026, 8, 31))
    assert split_period_predicate("train", OLD_DESIGN, 30) is not None  # train_start bound only
    frame = pl.DataFrame({"date": pl.date_range(D(2025, 12, 1), D(2026, 6, 1), "1d", eager=True)})
    assert frame.filter(split_period_predicate("train", OLD_DESIGN, 30))["date"].min() == D(
        2026, 1, 1
    )
    assert frame.filter(split_period_predicate("train", OLD_DESIGN, 30))["date"].max() == D(
        2026, 6, 1
    )
    assert frame.filter(split_period_predicate("validation", OLD_DESIGN, 30))["date"].max() == D(
        2026, 6, 1
    )


def test_resolved_test_end_caps_each_horizon_as_before():
    assert resolved_test_end(OLD_DESIGN, 30) == D(2026, 8, 31)
    assert resolved_test_end(OLD_DESIGN, 44) == D(2026, 8, 17)
    assert resolved_test_end(OLD_DESIGN, 60) == D(2026, 8, 1)


def test_feature_stats_cutoff_prefers_its_own_setting_and_falls_back_to_sealed_start():
    assert feature_stats_cutoff(NEW_DESIGN) == D(2026, 4, 21)
    assert feature_stats_cutoff(OLD_DESIGN) == D(2026, 7, 1)  # unchanged without the setting
    assert feature_stats_cutoff({"splits": {}}) is None


def test_a_zero_holdout_fraction_holds_no_drive_out():
    ids = [f"d{i}" for i in range(50)]
    assert sample_holdout_drive_ids(ids, holdout_fraction=0) == []
    assert len(sample_holdout_drive_ids(ids, holdout_fraction=0.2)) == 10
    frame = add_chronological_split(
        pl.DataFrame({"drive_id": ids, "date": [D(2026, 1, 2)] * 50}),
        train_end="2026-05-01",
        validation_end="2026-05-01",
    )
    out = apply_drive_level_holdout(frame, holdout_fraction=0, holdout_drive_ids=[])
    assert out["split"].unique().to_list() == ["train"]
    assert apply_drive_level_holdout(frame, holdout_fraction=0)["split"].unique().to_list() == [
        "train"
    ]


def test_gold_features_read_the_cutoff_from_the_model_config(tmp_path, monkeypatch):
    import pipelines.build_gold_features as bgf

    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"splits": {"sealed_start": "2026-07-01"}}))
    monkeypatch.setattr(bgf, "MODEL_CONFIG_PATH", path)
    assert bgf.feature_stats_cutoff() == D(2026, 7, 1)
    path.write_text(
        yaml.safe_dump(
            {"splits": {"sealed_start": "2026-07-01", "feature_stats_end": "2026-04-21"}}
        )
    )
    assert bgf.feature_stats_cutoff() == D(2026, 4, 21)
    assert bgf.sealed_start() == D(2026, 7, 1)  # the sealed window itself is unchanged


def test_first_dates_without_an_embargo_are_what_the_consumers_computed_before():
    """Old design: windows after validation start the day after `validation_end`, and the
    sealed test starts at `sealed_start` (fleet.overlaps_test_split, run_agent_models)."""
    assert first_date_after_validation(OLD_DESIGN) == D(2026, 5, 2)
    assert first_test_date(OLD_DESIGN) == D(2026, 7, 1)
    unsealed = _with(OLD_DESIGN, test_split="test")
    assert first_test_date(unsealed) == D(2026, 5, 2)


def test_first_dates_with_the_embargo_skip_it():
    assert first_date_after_validation(NEW_DESIGN) == D(2026, 7, 17)
    assert first_test_date(NEW_DESIGN) == D(
        2026, 7, 17
    )  # after sealed_start 07-01, so the embargo binds
    late_seal = _with(NEW_DESIGN, sealed_start="2026-08-01")
    assert first_test_date(late_seal) == D(2026, 8, 1)  # a later sealed_start binds instead
    assert first_test_date({"splits": {}}) is None


def test_the_agent_overlap_guard_uses_the_embargo():
    from src.agent.fleet import overlaps_test_split

    splits = NEW_DESIGN["splits"]
    assert not overlaps_test_split(D(2026, 7, 1), D(2026, 7, 16), splits)  # inside the embargo
    assert overlaps_test_split(D(2026, 7, 1), D(2026, 7, 17), splits)
    assert overlaps_test_split(None, None, splits)
    old = OLD_DESIGN["splits"]
    assert not overlaps_test_split(D(2026, 5, 2), D(2026, 6, 30), old)
    assert overlaps_test_split(D(2026, 6, 30), D(2026, 7, 1), old)
