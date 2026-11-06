# ruff: noqa: E501  (markdown table rows are written as single lines)
"""Deep dive into a Q1+Q2 -> Q3 training run (runbook 1.1.4).

    uv run python scripts/deep_dive_q3.py \\
        --run-dir frozen_runs/c459046e4f1645169151e7be25ff064d \\
        --scores data/tmp/lightgbm_scores_h30_h30_q1q2_q3.parquet \\
        --test-ids data/tmp/train_model_frame_fag3o60h/test_ids.parquet

Reads the run's saved validation and test scores (the combined two-stage score
per drive-day), its frozen thresholds, the test ids, the gold features and the
silver drive metadata, and answers four questions, per drive:

1. **Missed failures.** Which failing drives does each operating point miss,
   and do the ones Stage 1 never passes really show no SMART signal?
2. **False alarms.** Who are the healthy drives flagged at each operating
   point: still running, removed without a recorded failure, failed later, and
   which drive families?
3. **Stability over time.** Precision, recall, base rate and the Stage 1 pass
   rate week by week, on test and on validation.
4. **Validation against test.** How much of the precision gap is the base
   rate, the ranking, or the drive mix.

Nothing is re-chosen: every threshold is the frozen run's. Reading the test
scores again is a look at the test split, so one line is recorded in the
test-access log. Writes `deep_dive.json`, `tables.md` and `figures/` to `--out`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.models.access_log import _git_sha, record_test_access
from src.models.evaluation import recall_at_false_alarm_rate
from src.outputs import refuse_overwrite
from src.reporting.plots import plt, save_figure

CORE = [
    "reallocated_sector_count",
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",
    "command_timeout",
    "reallocation_event_count",
]
FEATURE_COLUMNS = [
    *CORE,
    "active_defect_total",
    "power_on_hours",
    "telemetry_coverage_30d",
    "feature_confidence",
    "model_family",
    "manufacturer",
]
#: Operating points read from the frozen run, strictest last.
POINT_ORDER = ("stage1", "warn", "cordon", "primary")
#: Missed-failure groups, from best caught to never seen by Stage 2.
GROUPS = (
    "caught at primary",
    "caught at cordon only",
    "caught at warn only",
    "passed Stage 1, below warn",
    "never passed Stage 1",
)


def load_thresholds(run_dir: Path) -> dict[str, float]:
    spec = json.loads((run_dir / "frozen_spec.json").read_text())
    two_stage = json.loads((run_dir / "two_stage.json").read_text())
    tiers = spec["action_tier_thresholds"]
    return {
        "stage1": float(two_stage["candidate_threshold"]),
        "warn": float(tiers["warn"]),
        "cordon": float(tiers["cordon"]),
        "primary": float(spec["drive_threshold"]),
    }


def drive_table(rows: pl.DataFrame, thresholds: dict[str, float]) -> pl.DataFrame:
    """One row per drive: failing or not, drive score (`drive_level_table`'s
    definition), first alert per operating point, and the SMART evidence
    inside the label window (all rows for a healthy drive)."""
    positive = pl.col("label") == 1
    window = pl.when(pl.col("failing")).then(positive).otherwise(pl.lit(True))
    rows = rows.sort("drive_id", "date").with_columns(failing=positive.any().over("drive_id"))
    core_max = pl.max_horizontal([pl.col(c).fill_null(0) for c in CORE])
    aggs: list[pl.Expr] = [
        pl.col("failing").first(),
        pl.col("score").filter(window).max().alias("drive_score"),
        pl.col("score").filter(~positive).max().alias("pre_window_max"),
        pl.len().alias("rows"),
        pl.col("date").max().alias("last_date"),
        pl.col("event_type").last(),
        pl.col("days_to_event").filter(positive).max().alias("window_days"),
        core_max.filter(window).max().alias("core_max"),
        (core_max.filter(window).max() - core_max.filter(window).min()).alias("core_increase"),
        pl.col("active_defect_total").filter(window).max().alias("active_defect_max"),
        pl.col("telemetry_coverage_30d").filter(window).mean().alias("coverage_mean"),
        pl.col("feature_confidence").filter(window).min().alias("confidence_min"),
        # Power-on hours, not the drive-age feature: that one counts from the first
        # day in the loaded data (2025-10-01), so it is not how old the drive is.
        (pl.col("power_on_hours").max() / (24 * 365.25)).alias("age_years"),
        pl.col("model_family").drop_nulls().first(),
        pl.col("manufacturer").drop_nulls().first(),
    ]
    for name, t in thresholds.items():
        aggs.append(
            pl.col("date").filter(window & (pl.col("score") >= t)).min().alias(f"first_{name}")
        )
        aggs.append(
            pl.col("days_to_event")
            .filter(window & (pl.col("score") >= t))
            .max()
            .alias(f"lead_{name}")
        )
    return rows.group_by("drive_id").agg(aggs).sort("drive_id")


def flags(drives: pl.DataFrame, thresholds: dict[str, float]) -> pl.DataFrame:
    return drives.with_columns(
        [(pl.col("drive_score") >= t).alias(f"flag_{n}") for n, t in thresholds.items()]
    )


def point_metrics(drives: pl.DataFrame, name: str) -> dict[str, Any]:
    f = drives[f"flag_{name}"].to_numpy()
    y = drives["failing"].to_numpy()
    caught, fa = int((f & y).sum()), int((f & ~y).sum())
    failing, healthy = int(y.sum()), int((~y).sum())
    return {
        "precision": caught / (caught + fa) if caught + fa else 0.0,
        "recall": caught / failing if failing else 0.0,
        "false_alarm_rate": fa / healthy if healthy else 0.0,
        "caught": caught,
        "false_alarms": fa,
        "failing": failing,
        "healthy": healthy,
    }


# --- 1. missed failures --------------------------------------------------------


def missed_failures(drives: pl.DataFrame) -> dict[str, Any]:
    failing = drives.filter(pl.col("failing")).with_columns(
        group=pl.when(pl.col("flag_primary"))
        .then(pl.lit(GROUPS[0]))
        .when(pl.col("flag_cordon"))
        .then(pl.lit(GROUPS[1]))
        .when(pl.col("flag_warn"))
        .then(pl.lit(GROUPS[2]))
        .when(pl.col("flag_stage1"))
        .then(pl.lit(GROUPS[3]))
        .otherwise(pl.lit(GROUPS[4]))
    )
    rows = []
    for g in GROUPS:
        part = failing.filter(pl.col("group") == g)
        n = part.height
        if not n:
            continue
        rows.append(
            {
                "group": g,
                "drives": n,
                "share": n / failing.height,
                "any_core_counter_nonzero": float((part["core_max"] > 0).mean()),
                "core_counter_increased_in_window": float((part["core_increase"] > 0).mean()),
                "active_defects_median": float(part["active_defect_max"].median() or 0),
                "telemetry_coverage_mean": float(part["coverage_mean"].mean() or 0),
                "age_years_median": float(part["age_years"].median() or 0),
                "alerted_at_warn_before_window": float(
                    (part["pre_window_max"].fill_null(-1.0) >= THRESHOLDS["warn"]).mean()
                ),
                "median_lead_days_at_own_point": _median_lead(part, g),
                "top_families": _top_share(part, "model_family", 4),
            }
        )
    no_signal = failing.filter((pl.col("core_max") == 0) & (pl.col("core_increase") == 0))
    by_family = (
        failing.group_by("model_family")
        .agg(
            pl.len().alias("failing"),
            pl.col("flag_stage1").mean().alias("stage1_recall"),
            pl.col("flag_warn").mean().alias("warn_recall"),
            pl.col("flag_primary").mean().alias("primary_recall"),
            (pl.col("core_max") == 0).mean().alias("no_core_signal_share"),
        )
        .sort("failing", descending=True)
        .head(10)
    )
    return {
        "failing_drives": failing.height,
        "groups": rows,
        "no_core_signal": {
            "drives": no_signal.height,
            "share_of_failing": no_signal.height / failing.height,
            "stage1_recall_among_them": float(no_signal["flag_stage1"].mean() or 0),
            "stage1_recall_among_the_rest": float(
                failing.filter(~pl.col("drive_id").is_in(no_signal["drive_id"]))[
                    "flag_stage1"
                ].mean()
                or 0
            ),
        },
        "by_family": by_family.to_dicts(),
    }


def _median_lead(part: pl.DataFrame, group: str) -> float | None:
    column = {
        GROUPS[0]: "lead_primary",
        GROUPS[1]: "lead_cordon",
        GROUPS[2]: "lead_warn",
        GROUPS[3]: "lead_stage1",
    }.get(group)
    if column is None:
        return None
    values = part[column].drop_nulls()
    return float(values.median()) if values.len() else None


def _top_share(part: pl.DataFrame, column: str, n: int) -> list[dict[str, Any]]:
    counts = part.group_by(column).agg(pl.len().alias("n")).sort("n", descending=True).head(n)
    return [{"name": r[column], "share": r["n"] / part.height} for r in counts.to_dicts()]


# --- 2. false alarms ------------------------------------------------------------


def false_alarms(drives: pl.DataFrame, metadata: pl.DataFrame) -> dict[str, Any]:
    healthy = drives.filter(~pl.col("failing")).join(
        metadata.select("drive_id", "failure_date", "last_seen_date"), on="drive_id", how="left"
    )
    fleet_family = healthy.group_by("model_family").agg(pl.len().alias("fleet"))
    out: dict[str, Any] = {"healthy_drives": healthy.height, "points": {}}
    for name in POINT_ORDER:
        flagged = healthy.filter(pl.col(f"flag_{name}"))
        if not flagged.height:
            continue
        failed_later = flagged.filter(pl.col("failure_date").is_not_null())
        outcomes = (
            flagged.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True)
        )
        family = (
            flagged.group_by("model_family")
            .agg(pl.len().alias("flagged"))
            .join(fleet_family, on="model_family")
            .with_columns(
                share=pl.col("flagged") / flagged.height,
                fleet_share=pl.col("fleet") / healthy.height,
            )
            .with_columns(lift=pl.col("share") / pl.col("fleet_share"))
            .sort("flagged", descending=True)
            .head(8)
        )
        out["points"][name] = {
            "flagged_healthy": flagged.height,
            "outcomes": {r["event_type"]: r["n"] for r in outcomes.to_dicts()},
            "failed_later": failed_later.height,
            "failed_later_days_after_first_alert_median": (
                float(
                    (failed_later["failure_date"] - failed_later[f"first_{name}"])
                    .dt.total_days()
                    .median()
                )
                if failed_later.height
                else None
            ),
            "any_core_counter_nonzero": float((flagged["core_max"] > 0).mean()),
            "core_counter_increased": float((flagged["core_increase"] > 0).mean()),
            "fleet_any_core_counter_nonzero": float((healthy["core_max"] > 0).mean()),
            "age_years_median": float(flagged["age_years"].median() or 0),
            "fleet_age_years_median": float(healthy["age_years"].median() or 0),
            "families": family.to_dicts(),
        }
    return out


# --- 3. stability over time -------------------------------------------------------


def weekly(rows: pl.DataFrame, start: dt.date, thresholds: dict[str, float]) -> list[dict]:
    """Per 7-day block from `start`: drives seen, failing drives (a positive row
    in the block), and drive metrics within the block at each operating point."""
    rows = rows.with_columns(
        week=((pl.col("date") - pl.lit(start)).dt.total_days() // 7).cast(pl.Int32)
    )
    positive = pl.col("label") == 1
    per = (
        rows.with_columns(failing=positive.any().over(["week", "drive_id"]))
        .group_by("week", "drive_id")
        .agg(
            pl.col("failing").first(),
            pl.col("score")
            .filter(pl.when(pl.col("failing")).then(positive).otherwise(pl.lit(True)))
            .max()
            .alias("s"),
            pl.col("score").max().alias("any_max"),
            pl.col("date").min().alias("d0"),
        )
    )
    out = []
    for (week,), part in sorted(per.group_by("week"), key=lambda kv: kv[0][0]):
        y = part["failing"].to_numpy()
        s = part["s"].to_numpy()
        entry: dict[str, Any] = {
            "week": int(week),
            "start": str(start + dt.timedelta(days=7 * int(week))),
            "drives": part.height,
            "failing": int(y.sum()),
            "base_rate": float(y.mean()),
            "stage1_pass_share": float((part["any_max"].to_numpy() >= thresholds["stage1"]).mean()),
        }
        for name in ("warn", "cordon", "primary"):
            f = s >= thresholds[name]
            caught, fa = int((f & y).sum()), int((f & ~y).sum())
            entry[name] = {
                "precision": caught / (caught + fa) if caught + fa else None,
                "recall": caught / y.sum() if y.sum() else None,
                "caught": caught,
                "false_alarms": fa,
            }
        out.append(entry)
    return out


# --- 4. validation against test ---------------------------------------------------


def restate(precision_recall_far: dict[str, float], base_rate: float) -> float:
    r, f = precision_recall_far["recall"], precision_recall_far["false_alarm_rate"]
    return r * base_rate / (r * base_rate + f * (1 - base_rate)) if r * base_rate else 0.0


def split_summary(drives: pl.DataFrame, label: str, days: int) -> dict[str, Any]:
    y = drives["failing"].to_numpy()
    s = drives["drive_score"].to_numpy()
    ids = np.arange(len(y))
    return {
        "split": label,
        "window_days": days,
        "drives": len(y),
        "failing": int(y.sum()),
        "base_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, s)),
        "points": {n: point_metrics(drives, n) for n in POINT_ORDER},
        "recall_at_far": {
            str(b): recall_at_false_alarm_rate(ids, y.astype(int), s, b) for b in (1e-4, 1e-3)
        },
        "families": _top_share(drives, "model_family", 6),
    }


def validation_vs_test(
    val: pl.DataFrame,
    test: pl.DataFrame,
    val_late: pl.DataFrame,
    test_new_ids: pl.Series,
) -> dict[str, Any]:
    v, t = split_summary(val, "validation", 121), split_summary(test, "test", 62)
    v_late = split_summary(val_late, "validation, last 62 days", 62)
    restated = {
        name: {
            "test_precision": t["points"][name]["precision"],
            "test_restated_at_validation_base_rate": restate(t["points"][name], v["base_rate"]),
            "validation_precision": v["points"][name]["precision"],
            "validation_restated_at_test_base_rate": restate(v["points"][name], t["base_rate"]),
        }
        for name in ("warn", "cordon", "primary")
    }
    new = test.filter(pl.col("drive_id").is_in(test_new_ids))
    old = test.filter(~pl.col("drive_id").is_in(test_new_ids))
    return {
        "validation": v,
        "validation_last_62_days": v_late,
        "test": t,
        "restated_precision": restated,
        "new_drives_in_test": {
            "drives": new.height,
            "failing": int(new["failing"].sum()),
            "primary": point_metrics(new, "primary") if new.height else None,
            "warn": point_metrics(new, "warn") if new.height else None,
            "older_drives_primary": point_metrics(old, "primary"),
            "older_drives_warn": point_metrics(old, "warn"),
        },
    }


# --- figures -----------------------------------------------------------------------


def figures(result: dict[str, Any], out: Path) -> list[Path]:
    paths = []
    groups = result["missed"]["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [g["group"] for g in groups]
    axes[0].barh(names, [g["drives"] for g in groups], color="#4878a8")
    for i, g in enumerate(groups):
        axes[0].text(g["drives"], i, f" {g['drives']} ({g['share']:.0%})", va="center", fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Failing test drives")
    axes[0].set_title("Where each failing drive stops (30-day model, Q3)")
    width = 0.38
    x = np.arange(len(groups))
    axes[1].bar(
        x - width / 2,
        [g["any_core_counter_nonzero"] for g in groups],
        width,
        label="any core counter > 0 in window",
    )
    axes[1].bar(
        x + width / 2,
        [g["core_counter_increased_in_window"] for g in groups],
        width,
        label="a core counter rose in window",
    )
    axes[1].set_xticks(
        x,
        [
            n.replace(", ", ",\n")
            .replace(" at ", "\nat ")
            .replace("passed Stage 1", "passed\nStage 1")
            for n in names
        ],
        fontsize=7,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share of drives in the group")
    axes[1].set_title("SMART evidence in the 30 days before failure")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    paths.append(save_figure(fig, out / "missed_failures.png"))

    pts = result["false_alarms"]["points"]
    order = [p for p in ("stage1", "warn", "cordon", "primary") if p in pts]
    kinds = sorted({k for p in order for k in pts[p]["outcomes"]})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(order))
    for kind in kinds:
        vals = np.array(
            [pts[p]["outcomes"].get(kind, 0) / pts[p]["flagged_healthy"] for p in order]
        )
        ax.bar(order, vals, bottom=bottom, label=kind)
        bottom += vals
    for i, p in enumerate(order):
        ax.text(i, 1.01, f"{pts[p]['flagged_healthy']:,}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of flagged healthy drives")
    ax.set_title("What the false alarms are, by operating point (count above each bar)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    paths.append(save_figure(fig, out / "false_alarm_outcomes.png"))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    for ax, key, title in (
        (axes[0], "test", "Q3 2026 test"),
        (axes[1], "validation", "Validation holdout"),
    ):
        weeks = result["weekly"][key]
        starts = [w["start"][5:] for w in weeks]
        for name, style in (("primary", "-o"), ("cordon", "-s"), ("warn", "-^")):
            ax.plot(starts, [w[name]["precision"] for w in weeks], style, label=f"{name} precision")
        ax2 = ax.twinx()
        ax2.plot(
            starts, [100 * w["base_rate"] for w in weeks], ":", color="grey", label="base rate (%)"
        )
        ax2.plot(
            starts,
            [100 * w["stage1_pass_share"] for w in weeks],
            "--",
            color="black",
            label="Stage 1 pass share (%)",
        )
        ax2.set_ylabel("%")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Precision")
        ax.set_title(f"{title}, week by week (thresholds fixed)")
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        lines = ax.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines[0] + lines2[0],
            lines[1] + lines2[1],
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.07, 1.0),
        )
    fig.tight_layout()
    paths.append(save_figure(fig, out / "weekly_stability.png"))

    gap = result["validation_vs_test"]["restated_precision"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(gap)
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, [gap[n]["validation_precision"] for n in names], w, label="validation")
    ax.bar(
        x,
        [gap[n]["test_restated_at_validation_base_rate"] for n in names],
        w,
        label="test, restated at validation base rate",
    )
    ax.bar(x + w, [gap[n]["test_precision"] for n in names], w, label="test")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Precision")
    ax.set_title("Validation against test: how much of the gap is the base rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(save_figure(fig, out / "validation_vs_test.png"))
    return paths


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def tables(result: dict[str, Any]) -> str:
    m, fa, gap = result["missed"], result["false_alarms"], result["validation_vs_test"]
    lines = [
        "# Deep dive: 30-day Q1+Q2 -> Q3 run",
        "",
        f"Run `{result['settings']['run_dir']}`, thresholds frozen: "
        + ", ".join(f"{k} {v:.3f}" for k, v in result["settings"]["thresholds"].items())
        + ". Per drive, Q3 2026 test (07-01..08-31) unless stated.",
        "",
        f"## 1. Missed failures ({m['failing_drives']:,} failing drives)",
        "",
        "| Group | Drives | Share | Any core counter > 0 | Counter rose in window | Active defects (median) | Telemetry coverage | Power-on age (years, median) | Warned before window | Median lead (days) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in m["groups"]:
        lead = g["median_lead_days_at_own_point"]
        lines.append(
            f"| {g['group']} | {g['drives']:,} | {g['share']:.1%} | {g['any_core_counter_nonzero']:.1%} "
            f"| {g['core_counter_increased_in_window']:.1%} | {g['active_defects_median']:.0f} "
            f"| {g['telemetry_coverage_mean']:.2f} | {g['age_years_median']:.1f} "
            f"| {g['alerted_at_warn_before_window']:.1%} | {'-' if lead is None else f'{lead:.0f}'} |"
        )
    ns = m["no_core_signal"]
    lines += [
        "",
        f"Failing drives with every core counter at zero throughout their window: "
        f"{ns['drives']:,} ({ns['share_of_failing']:.1%}). Stage 1 passes "
        f"{ns['stage1_recall_among_them']:.1%} of them against "
        f"{ns['stage1_recall_among_the_rest']:.1%} of the rest.",
        "",
        "By drive family (10 with the most failing drives):",
        "",
        "| Family | Failing | Stage 1 recall | Warn recall | Primary recall | No core signal |",
        "|---|---|---|---|---|---|",
    ]
    for r in m["by_family"]:
        lines.append(
            f"| {r['model_family']} | {r['failing']} | {_pct(r['stage1_recall'])} | {_pct(r['warn_recall'])} "
            f"| {_pct(r['primary_recall'])} | {_pct(r['no_core_signal_share'])} |"
        )
    lines += [
        "",
        f"## 2. False alarms ({fa['healthy_drives']:,} healthy drives)",
        "",
        "| Point | Flagged healthy | Still active | Removed without failure | Failed later | Any core counter > 0 (fleet) | Counter rose | Power-on age, years (fleet) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, p in fa["points"].items():
        o = p["outcomes"]
        n = p["flagged_healthy"]
        lines.append(
            f"| {name} | {n:,} | {o.get('still_active', 0) / n:.1%} "
            f"| {o.get('removed_without_failure', 0) / n:.1%} | {p['failed_later']:,} "
            f"| {p['any_core_counter_nonzero']:.1%} ({p['fleet_any_core_counter_nonzero']:.1%}) "
            f"| {p['core_counter_increased']:.1%} "
            f"| {p['age_years_median']:.1f} ({p['fleet_age_years_median']:.1f}) |"
        )
    for name in ("warn", "primary"):
        if name not in fa["points"]:
            continue
        lines += [
            "",
            f"Families among the {name} false alarms (lift = share of false alarms / share of healthy fleet):",
            "",
            "| Family | Flagged | Share | Fleet share | Lift |",
            "|---|---|---|---|---|",
        ]
        for r in fa["points"][name]["families"]:
            lines.append(
                f"| {r['model_family']} | {r['flagged']} | {r['share']:.1%} | {r['fleet_share']:.1%} | {r['lift']:.1f} |"
            )
    lines += [
        "",
        "## 3. Week by week (thresholds fixed; a drive is failing in a week if it has a positive row in it)",
        "",
        "| Split | Week from | Drives | Failing | Base rate | Stage 1 pass | Primary P / R | Cordon P / R | Warn P / R |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in ("test", "validation"):
        for w in result["weekly"][key]:
            lines.append(
                f"| {key} | {w['start']} | {w['drives']:,} | {w['failing']} | {w['base_rate']:.3%} "
                f"| {w['stage1_pass_share']:.2%} | {_pct(w['primary']['precision'])} / {_pct(w['primary']['recall'])} "
                f"| {_pct(w['cordon']['precision'])} / {_pct(w['cordon']['recall'])} "
                f"| {_pct(w['warn']['precision'])} / {_pct(w['warn']['recall'])} |"
            )
    lines += [
        "",
        "## 4. Validation against test",
        "",
        "| | Window | Drives | Failing | Base rate | AUROC | Recall at 0.01% / 0.1% false alarms | Primary P / R / false-alarm rate | Warn P / R |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in ("validation", "validation_last_62_days", "test"):
        s = gap[key]
        far = s["recall_at_far"]
        p, w = s["points"]["primary"], s["points"]["warn"]
        lines.append(
            f"| {s['split']} | {s['window_days']} d | {s['drives']:,} | {s['failing']:,} | {s['base_rate']:.3%} "
            f"| {s['auroc']:.3f} | {_pct(far['0.0001'].get('recall'))} / {_pct(far['0.001'].get('recall'))} "
            f"| {p['precision']:.1%} / {p['recall']:.1%} / {p['false_alarm_rate']:.3%} "
            f"| {w['precision']:.1%} / {w['recall']:.1%} |"
        )
    lines += [
        "",
        "| Point | Validation precision | Test precision | Test restated at the validation base rate | Validation restated at the test base rate |",
        "|---|---|---|---|---|",
    ]
    for name, r in gap["restated_precision"].items():
        lines.append(
            f"| {name} | {r['validation_precision']:.1%} | {r['test_precision']:.1%} "
            f"| {r['test_restated_at_validation_base_rate']:.1%} | {r['validation_restated_at_test_base_rate']:.1%} |"
        )
    nd = gap["new_drives_in_test"]
    if nd["primary"]:
        lines += [
            "",
            f"Drives first seen after 2026-05-01 (in no training or validation row): {nd['drives']:,}, "
            f"{nd['failing']} failing. Primary: precision {nd['primary']['precision']:.1%}, recall "
            f"{nd['primary']['recall']:.1%}, false-alarm rate {nd['primary']['false_alarm_rate']:.3%} "
            f"(older drives {nd['older_drives_primary']['precision']:.1%}, "
            f"{nd['older_drives_primary']['recall']:.1%}, {nd['older_drives_primary']['false_alarm_rate']:.3%}). "
            f"Warn recall {nd['warn']['recall']:.1%} against {nd['older_drives_warn']['recall']:.1%}.",
        ]
    return "\n".join(lines) + "\n"


THRESHOLDS: dict[str, float] = {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("data/gold/features/part.parquet"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/silver/drive_metadata/part.parquet")
    )
    parser.add_argument("--train-end", type=dt.date.fromisoformat, default=dt.date(2026, 5, 1))
    parser.add_argument("--out", type=Path, default=Path("data/audit/deep_dive_q3_h30"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Redraw the figures from an existing --out/deep_dive.json; reads no test data "
        "and records no test look.",
    )
    args = parser.parse_args(argv)
    if args.figures_only:
        result = json.loads((args.out / "deep_dive.json").read_text())
        paths = figures(result, args.out / "figures")
        print(f"redrew {len(paths)} figures in {args.out / 'figures'}")
        return
    refuse_overwrite(
        [args.out / "deep_dive.json", args.out / "tables.md"], overwrite=args.overwrite
    )

    THRESHOLDS.update(load_thresholds(args.run_dir))
    scores = pl.read_parquet(args.scores)
    metadata = pl.read_parquet(args.metadata)
    test_ids = pl.read_parquet(
        args.test_ids, columns=["drive_id", "date", "event_type", "days_to_event"]
    )

    def with_features(part: pl.DataFrame) -> pl.DataFrame:
        lo, hi = part["date"].min(), part["date"].max()
        feats = (
            pl.scan_parquet(args.features)
            .filter(pl.col("date").is_between(lo, hi))
            .select("drive_id", "date", *FEATURE_COLUMNS)
            .with_columns(pl.col("model_family", "manufacturer").cast(pl.Categorical))
            .collect(engine="streaming")
        )
        return part.join(feats, on=["drive_id", "date"], how="left")

    record_test_access(
        variant="deep_dive_q3",
        purpose="scripts/deep_dive_q3.py: missed failures, false alarms, weekly stability, validation vs test",
        extra={"run_dir": str(args.run_dir)},
    )
    test_rows = with_features(
        scores.filter(pl.col("split") == "test").join(test_ids, on=["drive_id", "date"], how="left")
    )
    val_rows = scores.filter(pl.col("split") == "validation").with_columns(
        event_type=pl.lit(None, dtype=pl.String), days_to_event=pl.lit(None, dtype=pl.Float64)
    )
    val_rows = with_features(val_rows)

    test_drives = flags(drive_table(test_rows, THRESHOLDS), THRESHOLDS)
    val_drives = flags(drive_table(val_rows, THRESHOLDS), THRESHOLDS)
    late_start = args.train_end - dt.timedelta(days=61)
    val_late = flags(
        drive_table(val_rows.filter(pl.col("date") >= late_start), THRESHOLDS), THRESHOLDS
    )
    new_ids = metadata.filter(pl.col("first_seen_date") > args.train_end)["drive_id"]

    result: dict[str, Any] = {
        "settings": {
            "run_dir": str(args.run_dir),
            "scores": str(args.scores),
            "thresholds": THRESHOLDS,
            "git_commit": _git_sha(),
        },
        "missed": missed_failures(test_drives),
        "false_alarms": false_alarms(test_drives, metadata),
        "weekly": {
            "test": weekly(test_rows, test_rows["date"].min(), THRESHOLDS),
            "validation": weekly(val_rows, val_rows["date"].min(), THRESHOLDS),
        },
        "validation_vs_test": validation_vs_test(val_drives, test_drives, val_late, new_ids),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "deep_dive.json").write_text(json.dumps(result, indent=2, default=str))
    (args.out / "tables.md").write_text(tables(result))
    paths = figures(result, args.out / "figures")
    print((args.out / "tables.md").read_text())
    print(f"written to {args.out} ({len(paths)} figures)")


if __name__ == "__main__":
    main()
