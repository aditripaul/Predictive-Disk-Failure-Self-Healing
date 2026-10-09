"""How many drive-days the per-model confidence fix moves across the 0.80
destructive-action floor, and for which drive models.

ADR 0002 fixed `attribute_coverage_factor` to count only the attributes each
drive MODEL reports. Before it, the denominator was every priority attribute,
so drives that do not report Seagate's smart_187/smart_188 were capped at
3/5 = 0.60 against a 0.80 floor and could never be cordoned, migrated or
drained. That ADR asks for this distribution to be inspected on a real build
before the new behaviour reaches a fleet, because it *unblocks* autonomous
destructive actions on a large part of it.

Both policies are computed on the same rows of one build, so this is a direct
comparison rather than two runs that might differ for other reasons:

    new (shipped) = telemetry_coverage_30d x recency_factor x (present among expected / expected)
    old (pre-fix) = telemetry_coverage_30d x recency_factor x (present / ALL priority attributes)

    uv run python scripts/check_confidence_distribution.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import yaml

DEFAULT_GOLD = Path("data/gold/features/part.parquet")
FEATURES_CONFIG = Path("configs/features.yaml")
DEFAULT_OUT = Path("data/audit/data_quality_reports/confidence_distribution.json")
#: configs/agent.yaml feature_confidence.min_confidence_for_destructive_action
FLOOR = 0.80


def _priority_attributes(config_path: Path) -> list[str]:
    return list(yaml.safe_load(config_path.read_text())["priority_smart_attributes"])


def compare(gold_path: Path, attributes: list[str], floor: float = FLOOR) -> dict:
    lf = pl.scan_parquet(gold_path)
    present = [a for a in attributes if a in lf.collect_schema().names()]
    missing = [a for a in attributes if a not in present]

    # The pre-fix policy: every priority attribute in the denominator.
    old_coverage = pl.sum_horizontal([pl.col(a).is_not_null() for a in present]) / len(present)
    old_conf = pl.col("telemetry_coverage_30d") * pl.col("recency_factor") * old_coverage

    rows = lf.select(
        [
            pl.col("drive_model"),
            pl.col("feature_confidence").alias("new_conf"),
            old_conf.alias("old_conf"),
            pl.col("attribute_coverage_factor"),
        ]
    )

    overall = rows.select(
        pl.len().alias("drive_days"),
        (pl.col("new_conf") >= floor).mean().alias("new_above_floor"),
        (pl.col("old_conf") >= floor).mean().alias("old_above_floor"),
        ((pl.col("new_conf") >= floor) & (pl.col("old_conf") < floor)).mean().alias("unblocked"),
        ((pl.col("new_conf") < floor) & (pl.col("old_conf") >= floor))
        .mean()
        .alias("newly_blocked"),
        pl.col("attribute_coverage_factor").min().alias("min_attribute_coverage"),
    ).collect()

    by_model = (
        rows.group_by("drive_model")
        .agg(
            pl.len().alias("drive_days"),
            (pl.col("old_conf") >= floor).mean().alias("old_above_floor"),
            (pl.col("new_conf") >= floor).mean().alias("new_above_floor"),
            pl.col("attribute_coverage_factor").median().alias("median_attribute_coverage"),
        )
        .filter(pl.col("drive_days") >= 10_000)
        .sort("drive_days", descending=True)
        .collect()
    )

    # feature_confidence is a product of three factors; when the share above
    # the floor is 0%, the useful question is which factor is binding.
    factors = ["telemetry_coverage_30d", "recency_factor", "attribute_coverage_factor"]
    factor_stats = (
        lf.select(
            [
                stat(pl.col(f)).alias(f"{f}__{name}")
                for f in factors
                for name, stat in (
                    ("median", lambda c: c.median()),
                    ("p95", lambda c: c.quantile(0.95)),
                    ("max", lambda c: c.max()),
                )
            ]
        )
        .collect()
        .to_dicts()[0]
    )
    by_factor = {
        f: {name: factor_stats[f"{f}__{name}"] for name in ("median", "p95", "max")}
        for f in factors
    }
    # A factor whose fleet-wide MAXIMUM is already below the floor makes the
    # floor unreachable on its own. But the factors can be anti-correlated -
    # recency reaches 1.0 only on a drive's first day, exactly when 30-day
    # coverage is 1/30 - so no factor need be individually capped for the
    # PRODUCT to be unreachable. The product's own maximum is the real answer.
    binding = [f for f, v in by_factor.items() if v["max"] is not None and v["max"] < floor]
    highest = lf.select(pl.col("feature_confidence").max().alias("m")).collect()["m"][0]

    return {
        "gold_path": str(gold_path),
        "factors": by_factor,
        "factors_below_floor_at_their_maximum": binding,
        "max_feature_confidence": highest,
        "floor_attainable_anywhere": bool(highest is not None and highest >= floor),
        "floor": floor,
        "priority_attributes": present,
        "priority_attributes_absent_from_gold": missing,
        "overall": overall.to_dicts()[0],
        "by_model": by_model.to_dicts(),
    }


def _print(report: dict) -> None:
    o = report["overall"]
    print(f"\n== feature_confidence against the {report['floor']} floor")
    print(f"  drive-days                  {o['drive_days']:,}")
    print(f"  above floor, pre-fix        {o['old_above_floor']:.1%}")
    print(f"  above floor, shipped        {o['new_above_floor']:.1%}")
    print(f"  UNBLOCKED by the fix        {o['unblocked']:.1%}")
    print(f"  newly blocked by the fix    {o['newly_blocked']:.2%}  (expected 0)")
    print(f"  min attribute_coverage      {o['min_attribute_coverage']}")
    if report["priority_attributes_absent_from_gold"]:
        print(f"  !! absent from gold: {report['priority_attributes_absent_from_gold']}")

    print("\n== the three factors it multiplies")
    print(f"{'factor':<30}{'median':>10}{'p95':>10}{'max':>10}")
    for name, v in report["factors"].items():
        print(f"{name:<30}{v['median']:>10.4f}{v['p95']:>10.4f}{v['max']:>10.4f}")
    print(
        f"{'feature_confidence (product)':<30}{'':>10}{'':>10}"
        f"{report['max_feature_confidence']:>10.4f}"
    )
    if not report["floor_attainable_anywhere"]:
        print(
            f"  !! NO drive-day anywhere reaches {report['floor']}: the agent could never"
            f" cordon, migrate or drain. The factors need not each be capped for their"
            f" product to be - check which ones sit below 1.0 above."
        )
    if report["factors_below_floor_at_their_maximum"]:
        for name in report["factors_below_floor_at_their_maximum"]:
            print(
                f"  !! {name} never reaches {report['floor']} anywhere in the fleet, so the"
                f" floor is unreachable regardless of the other two"
            )

    print(f"\n{'drive_model':<28}{'drive-days':>13}{'pre-fix':>10}{'shipped':>10}{'attr cov':>10}")
    for row in report["by_model"]:
        print(
            f"{row['drive_model'][:27]:<28}{row['drive_days']:>13,}"
            f"{row['old_above_floor']:>10.1%}{row['new_above_floor']:>10.1%}"
            f"{row['median_attribute_coverage']:>10.2f}"
        )
    print("\n  A model at 0% pre-fix and ~100% shipped is one the guardrail could never act on.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--config", type=Path, default=FEATURES_CONFIG)
    parser.add_argument("--floor", type=float, default=FLOOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.gold.exists():
        raise SystemExit(f"{args.gold} not found; run make build-features first")
    report = compare(args.gold, _priority_attributes(args.config), args.floor)
    _print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
