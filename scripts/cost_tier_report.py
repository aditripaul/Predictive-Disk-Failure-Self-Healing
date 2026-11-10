"""Lift tiers against cost tiers, in net value, on validation only.

The action tiers the agent uses are placed at lift targets that are placeholders
(configs/model.yaml threshold.action_tier_lift). The cost tiers (`tune_cost_tiers`)
place each tier where its action pays for itself, given how many unnecessary actions
one prevented failure is worth (`r`). This script puts the two side by side for one
scorer, so the operating point can be argued as a cost rather than a precision:

1. the ratios in force (`resolve_cost_ratios`: `threshold.cost_model` when filled in,
   otherwise the placeholder `threshold.action_cost_ratio`) and each tier's break-even
   marginal precision `1 / (1 + r)`;
2. per tier, the lift and the cost threshold on the validation drive table: flagged,
   caught, false alarms, precision, recall and net value `r * caught - false_alarms`
   (in actions of that tier; and in failures, when the cost model is filled in);
3. per lift tier, the ratio it implicitly assumes (`implied_cost_ratio`): the range of
   `r` for which its threshold is the cost-optimal cut;
4. with `--pit-observations`, the same two tier sets on the agent's point-in-time
   snapshots (`make chrono-pit-validation`), per snapshot.

Validation rows only: the score table is filtered to `split == "validation"` before
anything is materialised, and the point-in-time observations are validation snapshots.
No failure after the validation outcome window is read, so late hits (failures in days
31-60) are not credited here; on the chronological split those days fall in the test
period.

Both tier sets are tuned on the same validation drives they are reported on, so the
cost tiers' net value is in-sample by construction: the table shows where the cost
tiers would sit and what the lift tiers assume, not that one set beats the other.

    make chrono-cost-tiers        # the four chronological arms, placeholder or configured costs
    uv run python scripts/cost_tier_report.py --scores <score table> --score-column score \\
        --label <name> --out <file.md> [--pit-observations <observations.parquet>] \\
        [--failure-cost 1000 --action-cost warn=1 cordon=10 migrate=50 drain=200 \\
         --p-save warn=0.1 cordon=0.5 migrate=0.8 drain=0.9]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluation import drive_level_table  # noqa: E402
from src.models.threshold import (  # noqa: E402
    ACTION_TIERS,
    cost_ratios_from_costs,
    implied_cost_ratio,
    precision_targets_from_lift,
    resolve_cost_ratios,
    tune_action_tiers,
    tune_cost_tiers,
)

MODEL_CONFIG_PATH = Path("configs/model.yaml")
RULES = ("lift", "cost")


def load_validation_scores(path: Path, score_column: str) -> pl.DataFrame:
    """`drive_id, label, score` for the validation rows of a score table."""
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("split") == "validation")
        .select("drive_id", "label", pl.col(score_column).alias("score"))
        .collect()
    )
    if frame.is_empty():
        raise SystemExit(f"{path}: no validation rows")
    return frame


def _tier_kv(values: list[str] | None, name: str) -> dict[str, float] | None:
    if not values:
        return None
    out: dict[str, float] = {}
    for item in values:
        tier, _, value = item.partition("=")
        if tier not in ACTION_TIERS or not value:
            raise SystemExit(f"--{name} takes tier=value with tier in {ACTION_TIERS}, got {item!r}")
        out[tier] = float(value)
    missing = set(ACTION_TIERS) - set(out)
    if missing:
        raise SystemExit(f"--{name} is missing {sorted(missing)}")
    return out


def cost_inputs(
    threshold_cfg: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, float], str, dict[str, Any] | None]:
    """`(ratios, source, costs)`; `costs` is the cost model when one is in force, so net
    value can also be stated in failures."""
    action_cost = _tier_kv(args.action_cost, "action-cost")
    p_save = _tier_kv(args.p_save, "p-save")
    given = [args.failure_cost is not None, action_cost is not None, p_save is not None]
    if any(given):
        if not all(given):
            raise SystemExit("--failure-cost, --action-cost and --p-save go together")
        ratios = cost_ratios_from_costs(args.failure_cost, action_cost, p_save)
        return (
            ratios,
            "command line",
            {
                "failure_cost": args.failure_cost,
                "action_cost": action_cost,
                "p_save": p_save,
            },
        )
    ratios, source = resolve_cost_ratios(threshold_cfg)
    costs = threshold_cfg.get("cost_model") if source == "cost_model" else None
    return ratios, source, costs


def point(labels: np.ndarray, scores: np.ndarray, threshold: float | None, r: float) -> dict:
    if threshold is None:
        return {
            "threshold": None,
            "flagged": 0,
            "caught": 0,
            "false_alarms": 0,
            "precision": None,
            "recall": 0.0,
            "net": 0.0,
        }
    flagged = scores >= threshold
    caught = int((flagged & (labels == 1)).sum())
    false_alarms = int((flagged & (labels == 0)).sum())
    n = caught + false_alarms
    return {
        "threshold": float(threshold),
        "flagged": n,
        "caught": caught,
        "false_alarms": false_alarms,
        "precision": caught / n if n else None,
        "recall": caught / max(1, int(labels.sum())),
        "net": r * caught - false_alarms,
    }


def in_failures(net_actions: float, tier: str, costs: dict[str, Any] | None) -> float | None:
    """Net value in units of one unplanned failure: `action_cost / failure_cost` per
    action-unit of `r * caught - false_alarms`."""
    if not costs:
        return None
    return net_actions * float(costs["action_cost"][tier]) / float(costs["failure_cost"])


def pit_points(obs: pl.DataFrame, thresholds: dict[str, dict], ratios: dict[str, float]) -> dict:
    """Per snapshot means for each rule and tier on the point-in-time observations."""
    dates = obs["date"].n_unique()
    p = obs["p_fail"].to_numpy()
    y = obs["label"].to_numpy().astype(int)
    positives = int(y.sum())
    out: dict[str, dict] = {}
    for rule in RULES:
        out[rule] = {}
        for tier in ACTION_TIERS:
            thr = thresholds[rule][tier]
            flagged = p >= thr if thr is not None else np.zeros_like(y, dtype=bool)
            caught = int((flagged & (y == 1)).sum())
            fa = int((flagged & (y == 0)).sum())
            out[rule][tier] = {
                "flagged_per_snapshot": (caught + fa) / dates,
                "caught_per_snapshot": caught / dates,
                "precision": caught / (caught + fa) if caught + fa else None,
                "recall": caught / max(1, positives),
                "net_per_snapshot": (ratios[tier] * caught - fa) / dates,
            }
    return {
        "snapshots": int(dates),
        "drive_snapshots": int(len(y)),
        "positives": positives,
        "rules": out,
    }


def _pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:,.{digits}f}"


def _thr(x: float | None) -> str:
    return "never fires" if x is None else f"{x:.4f}"


def render(report: dict[str, Any]) -> str:
    costs = report["costs"]
    lines = [
        f"# Lift tiers against cost tiers: {report['label']}",
        "",
        "Written by `scripts/cost_tier_report.py` from validation rows only (score table "
        f"`{report['scores']}`, column `{report['score_column']}`; commit `{report['git_sha']}`). "
        "Both tier sets are tuned on these validation drives, so the cost tiers' net value is "
        "in-sample by construction: read this as where the cost tiers would sit and what the "
        "lift tiers assume, not as one set beating the other. Late hits (failures in days 31-60) "
        "are not credited; on the chronological split those days lie in the test period.",
        "",
        f"Validation: {report['drives']:,} drives, {report['failing_drives']:,} failing "
        f"({_pct(report['failure_rate'], 3)}).",
        "",
        f"## Ratios in force (source: {report['ratio_source']})",
        "",
        "`r` is how many unnecessary actions of that kind one prevented failure is worth; an "
        "action pays while the marginal precision of the drives it adds is at least `1 / (1 + r)`.",
        "",
    ]
    if costs:
        lines += [
            "| Tier | Action cost | P(save) | r | Break-even marginal precision |",
            "|---|---|---|---|---|",
        ]
        for t in ACTION_TIERS:
            r = report["ratios"][t]
            lines.append(
                f"| {t} | {costs['action_cost'][t]:g} | {costs['p_save'][t]:g} | {r:.1f} | "
                f"{_pct(1 / (1 + r))} |"
            )
        lines += ["", f"Failure cost: {costs['failure_cost']:g}.", ""]
    else:
        lines += ["| Tier | r | Break-even marginal precision |", "|---|---|---|"]
        for t in ACTION_TIERS:
            r = report["ratios"][t]
            lines.append(f"| {t} | {r:g} | {_pct(1 / (1 + r))} |")
        lines += [
            "",
            "These are the placeholder ratios. Fill in `threshold.cost_model` in "
            "`configs/model.yaml` (or pass `--failure-cost`, `--action-cost`, `--p-save`) to "
            "derive them from costs.",
            "",
        ]

    lines += [
        "## What each lift tier assumes",
        "",
        "The range of `r` for which the lift tier's threshold is the cost-optimal cut on the "
        "validation drives, and the marginal precision that implies. A lift tier whose "
        "implied `r` is below the ratio in force is stricter than its cost justifies (flagging "
        "more would pay); above it, looser; within 10%, about the same. A precision target "
        "usually lands on a jagged point just under the curve's convex hull, where no `r` is "
        "exactly optimal; its implied `r` is then read from the hull segment above it "
        f"({report['off_hull']} of {len(ACTION_TIERS)} tiers here).",
        "",
        "| Tier | Lift threshold | Implied r | Marginal precision at the cut | r in force "
        "| Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for t in ACTION_TIERS:
        imp = report["implied"][t]
        r_now = report["ratios"][t]
        if imp is None:
            lines.append(f"| {t} | never fires | - | - | {r_now:g} | - |")
            continue
        lo, hi = imp["r_low"], imp["r_high"]
        implied = _num(lo, 1) if lo == hi else f"{_num(lo, 1)} to {_num(hi, 1)}"
        mp = (
            _pct(imp["marginal_precision_high"])
            if lo == hi
            else f"{_pct(imp['marginal_precision_low'])} to {_pct(imp['marginal_precision_high'])}"
        )
        if hi < r_now / 1.1:
            verdict = "stricter than its cost"
        elif lo > r_now * 1.1:
            verdict = "looser than its cost"
        else:
            verdict = "about the same"
        lines.append(f"| {t} | {imp['threshold']:.4f} | {implied} | {mp} | {r_now:g} | {verdict} |")

    unit = " / net (failures)" if costs else ""
    lines += [
        "",
        "## Validation drive table",
        "",
        "Each tier read on its own: every drive at or above its threshold takes that action. "
        "Net value is `r x caught - false alarms`, in actions of that tier"
        + ("; in failures it is that times action cost / failure cost." if costs else "."),
        "",
        "| Tier | Rule | Threshold | Flagged | Caught | False alarms | Precision | Recall "
        f"| Net (actions){unit} |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in ACTION_TIERS:
        for rule in RULES:
            pt = report["validation"][rule][t]
            net = _num(pt["net"])
            if costs:
                net += f" / {_num(in_failures(pt['net'], t, costs), 1)}"
            lines.append(
                f"| {t} | {rule} | {_thr(pt['threshold'])} | {pt['flagged']:,} | "
                f"{pt['caught']:,} | "
                f"{pt['false_alarms']:,} | {_pct(pt['precision'])} | {_pct(pt['recall'])} | {net} |"
            )

    pit = report.get("pit")
    if pit:
        lines += [
            "",
            "## Point in time",
            "",
            f"{pit['snapshots']} validation snapshots, {pit['drive_snapshots']:,} drive-snapshots, "
            f"{pit['positives']:,} failing within 30 days (`{report['pit_observations']}`). Per "
            "snapshot means; snapshots overlap, so they are not additive. Precision and recall "
            "are per drive-snapshot, lower than per drive.",
            "",
            "| Tier | Rule | Flagged / snapshot | Caught / snapshot | Precision | Recall "
            "| Net / snapshot |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in ACTION_TIERS:
            for rule in RULES:
                q = pit["rules"][rule][t]
                lines.append(
                    f"| {t} | {rule} | {_num(q['flagged_per_snapshot'], 1)} | "
                    f"{_num(q['caught_per_snapshot'], 1)} | {_pct(q['precision'])} | "
                    f"{_pct(q['recall'])} | {_num(q['net_per_snapshot'], 1)} |"
                )

    check = report.get("threshold_check")
    if check:
        lines += ["", "## Cross-check", "", check]
    return "\n".join(lines) + "\n"


def cross_check(path: Path, lift: dict[str, float | None]) -> str:
    """Compare the recomputed lift tiers with a thresholds file's."""
    saved = json.loads(path.read_text())
    saved = saved.get("action_tiers", saved)
    diffs = []
    for t in ACTION_TIERS:
        a, b = saved.get(t), lift[t]
        if (a is None) != (b is None):
            diffs.append(f"{t}: saved {a}, recomputed {b}")
        elif a is not None and abs(a - b) > 1e-9:
            diffs.append(f"{t}: saved {a:.6f}, recomputed {b:.6f}")
    if diffs:
        return f"Lift tiers differ from `{path}`: " + "; ".join(diffs) + "."
    return f"Recomputed lift tiers match `{path}` exactly."


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--thresholds", type=Path, help="thresholds file to cross-check the lift tiers"
    )
    parser.add_argument("--pit-observations", type=Path)
    parser.add_argument("--failure-cost", type=float)
    parser.add_argument("--action-cost", nargs="+", metavar="TIER=COST")
    parser.add_argument("--p-save", nargs="+", metavar="TIER=P")
    parser.add_argument(
        "--out", type=Path, required=True, help="markdown; a .json is written beside it"
    )
    args = parser.parse_args(argv)

    threshold_cfg = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["threshold"]
    ratios, source, costs = cost_inputs(threshold_cfg, args)

    frame = load_validation_scores(args.scores, args.score_column)
    labels, scores = drive_level_table(
        frame["drive_id"].to_numpy(), frame["label"].to_numpy(), frame["score"].to_numpy()
    )
    del frame
    ids = np.arange(len(labels))  # one row per drive already
    rate = float(labels.mean())
    lift = tune_action_tiers(
        ids,
        labels,
        scores,
        precision_targets=precision_targets_from_lift(threshold_cfg["action_tier_lift"], rate),
    )
    cost = tune_cost_tiers(ids, labels, scores, cost_ratios=ratios)
    thresholds = {"lift": lift, "cost": cost}

    report: dict[str, Any] = {
        "label": args.label,
        "scores": str(args.scores),
        "score_column": args.score_column,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "drives": int(len(labels)),
        "failing_drives": int(labels.sum()),
        "failure_rate": rate,
        "ratio_source": source,
        "ratios": ratios,
        "costs": costs,
        "thresholds": thresholds,
        "validation": {
            rule: {t: point(labels, scores, thresholds[rule][t], ratios[t]) for t in ACTION_TIERS}
            for rule in RULES
        },
        "implied": {
            t: None if lift[t] is None else implied_cost_ratio(labels, scores, lift[t])
            for t in ACTION_TIERS
        },
    }
    report["off_hull"] = sum(
        1 for imp in report["implied"].values() if imp is not None and not imp["on_hull"]
    )
    if args.thresholds:
        report["threshold_check"] = cross_check(args.thresholds, lift)
    if args.pit_observations:
        obs = pl.read_parquet(args.pit_observations, columns=["date", "p_fail", "label"])
        report["pit_observations"] = str(args.pit_observations)
        report["pit"] = pit_points(obs, thresholds, ratios)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(report))
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
