"""Paired comparison of models trained for different label horizons
(`scripts/compare_horizons.py`, docs/q3_horizon_and_cascade_run.md).

A longer horizon marks more drive-days positive, so precision and AUPRC rise
even when the model ranks drives no better. The comparison is therefore made
where the base rate does not enter: drive AUROC and recall at a matched
false-alarm rate (a share of HEALTHY drives). Every model is scored against
every label definition, on the same drive-days (the window all horizons can
observe), so each comparison is paired: same drives, same label, two models.

Drive scoring is `src/models/evaluation.py::drive_level_table`'s throughout.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.models.evaluation import (
    drive_level_table,
    recall_at_false_alarm_rate,
    top_k_from_drive_table,
)
from src.models.uncertainty import paired_flag_difference_ci

#: Healthy-drive false-alarm rates the horizons are matched at.
DEFAULT_BUDGETS = (1e-3, 5e-3)


def common_frame(
    scores: Mapping[int, pl.DataFrame],
    labels: pl.DataFrame,
    *,
    start: dt.date,
    end: dt.date,
) -> pl.DataFrame:
    """One row per drive-day every model scored inside [start, end], with
    `score_<h>` per model and `label_<h>` per label horizon; rows where any
    label is unobserved are dropped, so every comparison uses the same rows.

    `scores[h]`: drive_id, date, score. `labels`: drive_id, date, horizon_days,
    label, days_to_event (one row per drive-day per horizon)."""
    horizons = sorted(scores)
    frame: pl.DataFrame | None = None
    for h in horizons:
        part = (
            scores[h]
            .filter(pl.col("date").is_between(start, end))
            .select("drive_id", "date", pl.col("score").alias(f"score_{h}"))
        )
        frame = part if frame is None else frame.join(part, on=["drive_id", "date"], how="inner")
    assert frame is not None
    window = labels.filter(
        pl.col("date").is_between(start, end) & pl.col("horizon_days").is_in(horizons)
    )
    wide = window.pivot(on="horizon_days", index=["drive_id", "date"], values="label")
    wide = wide.rename({str(h): f"label_{h}" for h in horizons})
    to_event = window.group_by("drive_id", "date").agg(pl.col("days_to_event").first())
    frame = frame.join(wide, on=["drive_id", "date"], how="inner").join(
        to_event, on=["drive_id", "date"], how="left"
    )
    return frame.drop_nulls([f"label_{h}" for h in horizons]).sort("drive_id", "date")


def _first_alert_lead_days(
    frame: pl.DataFrame, label_col: str, score_col: str, threshold: float
) -> dict[str, Any]:
    """Days from a failing drive's first alert inside its label window to its
    failure, over the failing drives that alerted (the caught ones)."""
    alerts = (
        frame.filter((pl.col(label_col) == 1) & (pl.col(score_col) >= threshold))
        .group_by("drive_id")
        .agg(pl.col("days_to_event").max().alias("lead"))
    )["lead"].drop_nulls()
    if alerts.is_empty():
        return {"caught": 0, "median_days": None, "mean_days": None}
    leads = alerts.to_numpy().astype(np.float64)
    return {
        "caught": len(leads),
        "median_days": float(np.median(leads)),
        "mean_days": float(np.mean(leads)),
    }


def _auroc_bootstrap(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, *, n_boot: int, seed: int
) -> dict[str, Any]:
    """AUROC(B) - AUROC(A) on the same resampled drives."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        if labels[pick].min() == labels[pick].max():
            continue
        diffs.append(
            roc_auc_score(labels[pick], scores_b[pick])
            - roc_auc_score(labels[pick], scores_a[pick])
        )
    point = roc_auc_score(labels, scores_b) - roc_auc_score(labels, scores_a)
    return {
        "difference": float(point),
        "difference_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "n_boot": len(diffs),
    }


def compare_horizons(
    frame: pl.DataFrame,
    horizons: Sequence[int],
    *,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    top_k: Sequence[int] = (),
    n_boot: int = 1000,
    n_boot_auroc: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Every model (`score_<h>`) against every label definition (`label_<h>`),
    with paired differences against the shortest horizon's model."""
    horizons = sorted(horizons)
    reference = horizons[0]
    drive_ids = frame["drive_id"].to_numpy()
    out: dict[str, Any] = {"horizons": horizons, "reference": reference, "labels": {}}
    for label_h in horizons:
        y = frame[f"label_{label_h}"].to_numpy()
        tables = {
            m: drive_level_table(drive_ids, y, frame[f"score_{m}"].to_numpy()) for m in horizons
        }
        drive_labels = tables[reference][0]
        unique_drives = np.arange(len(drive_labels))
        per_model: dict[int, dict[str, Any]] = {}
        for m in horizons:
            labels_m, scores_m = tables[m]
            assert np.array_equal(labels_m, drive_labels)
            at_budget = []
            for budget in budgets:
                point = recall_at_false_alarm_rate(unique_drives, labels_m, scores_m, budget)
                if point.get("reachable"):
                    flagged = scores_m >= point["threshold"]
                    caught = int((flagged & (labels_m == 1)).sum())
                    alarms = int(flagged.sum())
                    point["precision"] = caught / alarms if alarms else 0.0
                    point["caught"] = caught
                    point["false_alarms"] = alarms - caught
                    point["lead_time"] = _first_alert_lead_days(
                        frame, f"label_{label_h}", f"score_{m}", point["threshold"]
                    )
                at_budget.append(point)
            per_model[m] = {
                "auroc": float(roc_auc_score(labels_m, scores_m)),
                "at_false_alarm_rate": at_budget,
                "top_k": top_k_from_drive_table(labels_m, scores_m, top_k) if top_k else [],
            }
            if m != reference:
                versus: dict[str, Any] = {
                    "auroc": _auroc_bootstrap(
                        drive_labels,
                        tables[reference][1],
                        scores_m,
                        n_boot=n_boot_auroc,
                        seed=seed,
                    ),
                    "at_false_alarm_rate": [],
                }
                for i, budget in enumerate(budgets):
                    ref_point = per_model[reference]["at_false_alarm_rate"][i]
                    point = at_budget[i]
                    if ref_point.get("reachable") and point.get("reachable"):
                        versus["at_false_alarm_rate"].append(
                            {
                                "budget": budget,
                                **paired_flag_difference_ci(
                                    drive_labels,
                                    tables[reference][1] >= ref_point["threshold"],
                                    scores_m >= point["threshold"],
                                    n_boot=n_boot,
                                    seed=seed,
                                ),
                            }
                        )
                per_model[m]["versus_reference"] = versus
        out["labels"][label_h] = {
            "drives": len(drive_labels),
            "failing_drives": int(drive_labels.sum()),
            "base_rate": float(drive_labels.mean()),
            "row_positive_rate": float(y.mean()),
            "models": per_model,
        }
    return out


def comparison_tables(result: dict[str, Any], settings: dict[str, Any]) -> str:
    """Markdown for the runbook and docs/q3_horizon_and_cascade_run.md."""
    ref = result["reference"]
    lines = [
        "# Horizon comparison (test, drive level, paired)",
        "",
        f"Window {settings['start']} .. {settings['end']} (rows every horizon can observe). "
        f"Each model scored against each label definition on the same drive-days. Paired "
        f"differences are against the {ref}-day model ({settings['n_boot']} drive resamples for "
        f"recall, {settings['n_boot_auroc']} for AUROC, 95%). False-alarm rate = share of "
        "healthy drives flagged; lead = days from a caught drive's first alert to its failure.",
    ]
    for label_h, block in result["labels"].items():
        lines += [
            "",
            f"## Scored against {label_h}-day labels: {block['failing_drives']:,} failing of "
            f"{block['drives']:,} drives (base rate {block['base_rate']:.3%})",
            "",
            "| Model | AUROC | AUROC gap vs ref [95%] | Budget | Recall | Recall gap vs ref [95%] "
            "| Precision | Caught / false alarms | Median lead (days) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for m, model in block["models"].items():
            versus = model.get("versus_reference", {})
            auroc_gap = versus.get("auroc")
            gap_cell = (
                f"{auroc_gap['difference']:+.3f} [{auroc_gap['difference_ci'][0]:+.3f}, "
                f"{auroc_gap['difference_ci'][1]:+.3f}]"
                if auroc_gap
                else "ref"
            )
            recall_gaps = {g["budget"]: g for g in versus.get("at_false_alarm_rate", [])}
            for point in model["at_false_alarm_rate"]:
                budget = point["false_alarm_rate_budget"]
                if not point.get("reachable"):
                    lines.append(f"| {m}-day | | | {budget:.1%} | unreachable | | | | |")
                    continue
                gap = recall_gaps.get(budget)
                recall_gap = (
                    f"{gap['recall_difference']:+.1%} [{gap['recall_difference_ci'][0]:+.1%}, "
                    f"{gap['recall_difference_ci'][1]:+.1%}]"
                    if gap
                    else ("ref" if m == ref else "-")
                )
                lead = point["lead_time"]["median_days"]
                lines.append(
                    f"| {m}-day | {model['auroc']:.3f} | {gap_cell} | {budget:.1%} | "
                    f"{point['recall']:.1%} | {recall_gap} | {point['precision']:.1%} | "
                    f"{point['caught']:,} / {point['false_alarms']:,} | "
                    f"{'-' if lead is None else f'{lead:.0f}'} |"
                )
        if any(model["top_k"] for model in block["models"].values()):
            ks = [row["k"] for row in next(iter(block["models"].values()))["top_k"]]
            lines += [
                "",
                "| Model | " + " | ".join(f"Top {k:,} P / R" for k in ks) + " |",
                "|---|" + "---|" * len(ks),
            ]
            for m, model in block["models"].items():
                cells = [f"{row['precision']:.1%} / {row['recall']:.1%}" for row in model["top_k"]]
                lines.append(f"| {m}-day | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
