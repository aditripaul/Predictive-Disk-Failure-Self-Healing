"""Does per-family recalibration beat one global threshold, out of sample?

Fits a shrunk per-family calibration (src/agent/family_calibration.py) on
point-in-time snapshots from the VALIDATION window, and judges it on the TEST
window's snapshots against the raw score, at matched false-alarm rate with a
paired drive-cluster bootstrap. Nothing here is fitted on, tuned on or chosen
by looking at test data: tau is picked by a time split inside the fitting
window.

    # 1. snapshots for fitting - validation window, separate output files
    make agent-pit ARGS="--start-date 2026-02-15 --end-date 2026-03-31 \\
        --out data/audit/agent_pit_validation.json \\
        --observations-out data/audit/agent_pit_validation_observations.parquet"
    # 2. the test snapshots already exist from `make agent-pit`
    uv run python scripts/evaluate_family_calibration.py

The raw score is the right baseline: a fleet-wide recalibration is monotone,
so it ranks exactly like the raw score. Any difference comes from the family
offsets reordering drives across families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.agent.family_calibration import FamilyCalibrator, select_tau
from src.agent.point_in_time import DEFAULT_FAR_BUDGETS, compare_scorings
from src.models.access_log import record_test_access

DEFAULT_FIT = Path("data/audit/agent_pit_validation_observations.parquet")
DEFAULT_EVAL = Path("data/audit/agent_point_in_time_observations.parquet")
DEFAULT_OUT = Path("data/audit/family_calibration.json")
#: The pooled cordon operating point sits at ~1.2e-3; 2e-3 brackets it.
BUDGETS = (*DEFAULT_FAR_BUDGETS[:4], 2e-3, DEFAULT_FAR_BUDGETS[4])


def check_windows(fit: pl.DataFrame, evaluate: pl.DataFrame) -> None:
    """Refuse unless every fitting snapshot precedes every evaluation one.
    Both files come from the same pipeline and are easy to swap; fitting on
    the test window would make the comparison meaningless."""
    if fit.is_empty() or evaluate.is_empty():
        raise SystemExit("a snapshot file is empty")
    fit_last = max(fit["date"].to_list())
    eval_first = min(evaluate["date"].to_list())
    if not fit_last < eval_first:
        raise SystemExit(
            f"fitting snapshots run to {fit_last} but evaluation starts {eval_first}: "
            "the fit must come entirely before the test window"
        )


def evaluate(
    fit: pl.DataFrame,
    test: pl.DataFrame,
    *,
    family_column: str = "drive_model",
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    check_windows(fit, test)
    for name, frame in (("fitting", fit), ("evaluation", test)):
        if family_column not in frame.columns:
            raise SystemExit(f"{family_column!r} is not in the {name} snapshots")

    tau, tau_scores = select_tau(fit, family_column=family_column)
    # tau = 0 means the family offsets did not help even inside the fitting
    # window; a near-zero prior width reproduces the fleet-only calibration.
    calibrator = FamilyCalibrator(family_column=family_column, tau=max(tau, 1e-6)).fit(fit)
    calibrated = calibrator.transform(test)
    failed = test["label"].to_numpy() == 1
    comparison = compare_scorings(
        test["drive_id"],
        failed,
        test["p_fail"].to_numpy(),
        calibrated,
        budgets=BUDGETS,
        n_boot=n_boot,
        seed=seed,
    )
    unseen = sorted(set(test[family_column].drop_nulls().unique()) - set(calibrator.offsets))
    return {
        "family_column": family_column,
        "fit_dates": [str(fit["date"].min()), str(fit["date"].max())],
        "eval_dates": [str(test["date"].min()), str(test["date"].max())],
        "tau": tau,
        "tau_selection": tau_scores,
        "calibrator": calibrator.to_dict(),
        "families_unseen_in_fit": unseen,
        "comparison": comparison,
    }


def print_result(result: dict) -> None:
    fit_from, fit_to = result["fit_dates"]
    eval_from, eval_to = result["eval_dates"]
    print(
        f"\n== per-family recalibration, fitted {fit_from}..{fit_to}, judged {eval_from}..{eval_to}"
    )
    print("\n1. prior width chosen inside the fitting window (later dates predicted from earlier)")
    for row in result["tau_selection"]:
        mark = "  <- chosen" if row["tau"] == result["tau"] else ""
        label = "fleet only" if row["tau"] == 0 else f"tau={row['tau']}"
        print(f"   {label:<12} log-loss {row['log_loss']:.6f}{mark}")
    if result["tau"] == 0:
        print("   Family offsets did not improve prediction even in the fitting window.")

    cal = result["calibrator"]
    offsets = cal["offsets"]
    failures = cal["failures_by_family"]
    top = sorted(offsets, key=lambda f: -abs(offsets[f]))[:12]
    print("\n2. largest family offsets (odds of failure at a given score, vs the fleet)")
    print(f"   {'drive model':<24}{'failures':>9}{'offset':>9}{'odds x':>9}")
    for family in top:
        print(
            f"   {family:<24}{failures.get(family, 0):>9}{offsets[family]:>+9.2f}"
            f"{np.exp(offsets[family]):>9.2f}"
        )
    if result["families_unseen_in_fit"]:
        print(
            f"   {len(result['families_unseen_in_fit'])} test families unseen in the fit "
            "use the fleet calibration"
        )

    print("\n3. on the test snapshots, at matched false-alarm rate (paired, drive-clustered)")
    print(
        f"   {'FAR':>8}{'raw recall':>12}{'family':>9}{'raw prec':>10}{'family':>9}"
        f"{'difference (95% CI)':>26}"
    )
    for row in result["comparison"]:
        lo, hi = row["difference_ci"]
        print(
            f"   {row['false_alarm_budget']:>8.0e}{row['baseline_recall']:>12.1%}"
            f"{row['candidate_recall']:>9.1%}{row['baseline_precision']:>10.1%}"
            f"{row['candidate_precision']:>9.1%}{row['difference']:>+11.1%} [{lo:+.1%},{hi:+.1%}]"
        )
    print("   A recall gain whose interval clears zero is real; one that straddles it is not.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--family-column", default="drive_model")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    for path in (args.fit, args.eval):
        if not path.exists():
            raise SystemExit(f"{path} not found; see this script's docstring for how to make it")

    fit, test = pl.read_parquet(args.fit), pl.read_parquet(args.eval)
    result = evaluate(fit, test, family_column=args.family_column, n_boot=args.bootstrap)
    record_test_access(
        variant="family_calibration",
        purpose="per-family recalibration fitted on validation snapshots, judged on test",
        extra={
            "fit_dates": result["fit_dates"],
            "eval_dates": result["eval_dates"],
            "tau": result["tau"],
        },
    )
    print_result(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
