"""Search the weights of a train-merged rank blend on VALIDATION; no retraining.

    make tune-blend-weights PACK_DIR=<packed scores dir> OUT_DIR=<dir>

Reads a packed score table (`make pack-merged-scores`; the frozen run's is under
frozen_runs/), ranks each member against its own validation scores, then scores every weight
vector on a coarse simplex grid by the mean drive-level recall at false-alarm rates 1e-4,
2e-4 and 5e-4 (src/models/blend_weights.py). Nothing here reads the test split unless
--evaluate-test is given, which spends ONE recorded look at test on the recommended weights.

A vector is only eligible if it beats equal weights on the full validation set AND on both
halves of it (split by date). Among the eligible ones within one bootstrap standard error of
the best, the closest to equal weights is recommended, and it is called an improvement only if
the paired drive bootstrap's 95% interval for its gain over equal weights excludes zero.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.models.access_log import record_test_access
from src.models.blend_weights import (
    FAR_BUDGETS,
    DriveRows,
    average_precision,
    bootstrap_difference,
    objective,
    pipeline_criterion,
    recall_at_far,
    simplex_grid,
)
from src.models.ensemble import rank_blend, validation_rank
from src.models.score_pack import MANIFEST, rebuild
from src.outputs import refuse_overwrite
from src.reporting.merged_tables import model_table

MODEL_CONFIG_PATH = Path("configs/model.yaml")
#: Recall floors at which each vector's best validation precision is also recorded.
RECALL_FLOORS = (0.05, 0.075, 0.10)


def _evaluate(rows_ctx: dict, w: np.ndarray, criterion: dict) -> dict:
    """Validation metrics of weight vector `w` on the full set and on each half."""
    budgets, floors = rows_ctx["budgets"], rows_ctx["floors"]
    scores = rows_ctx["full"][0].drive_scores(rows_ctx["full"][1] @ w)
    labels = rows_ctx["full"][0].labels
    half = {
        name: objective(rows.labels, rows.drive_scores(arranged @ w), budgets)
        for name, (rows, arranged) in rows_ctx["halves"].items()
    }
    pick = pipeline_criterion(labels, scores, **criterion)
    by_floor = {
        str(floor): pipeline_criterion(
            labels, scores, target_precision=criterion["target_precision"], recall_floor=floor
        )
        for floor in floors
    }
    return {
        "weights": [round(float(x), 4) for x in w],
        "objective": objective(labels, scores, budgets),
        "recall_at_far": recall_at_far(labels, scores, budgets),
        "average_precision": average_precision(labels, scores),
        "half_a": half["a"],
        "half_b": half["b"],
        "pipeline_precision": pick["precision"],
        "pipeline_recall": pick["recall"],
        "precision_at_recall_floor": {
            floor: [round(point["precision"], 4), round(point["recall"], 4)]
            for floor, point in by_floor.items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pack_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--blend", default="lgbm_lstm_aft_tcn")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--far-budgets",
        type=float,
        nargs="+",
        default=list(FAR_BUDGETS),
        help="false-alarm rates the search maximises recall at (mean). Default: the low-alert "
        "regime 1e-4 2e-4 5e-4; for a recall-first goal try 1e-3 2e-3 5e-3 1e-2",
    )
    parser.add_argument(
        "--recall-floors",
        type=float,
        nargs="+",
        default=list(RECALL_FLOORS),
        help="recall floors at which each vector's best validation precision is recorded",
    )
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        action="append",
        metavar="W",
        help="score these weights (in member order; any non-negative values, normalised to sum "
        "to 1) against equal weights on validation instead of searching the grid; repeatable",
    )
    parser.add_argument("--evaluate-test", action="store_true", help="spend one recorded test look")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary_path = args.out_dir / ("custom.json" if args.weights else "summary.json")
    refuse_overwrite([summary_path], overwrite=args.overwrite)

    config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    threshold = config["threshold"]
    criterion = {
        "target_precision": threshold["target_precision"],
        "recall_floor": threshold["target_recall_range"][0],
    }
    manifest = json.loads((args.pack_dir / MANIFEST).read_text())
    members = manifest["ensembles"][args.blend]

    t0 = time.perf_counter()
    table = rebuild(args.pack_dir)
    val = table.filter(pl.col("split") == "validation")
    loaded = time.perf_counter() - t0
    print(f"loaded {table.height:,} rows ({val.height:,} validation) in {loaded:.0f}s")
    reference = {m: val[m].to_numpy() for m in members}
    ranks = np.column_stack([validation_rank(reference[m], reference[m]) for m in members])
    equal = np.full(len(members), 1 / len(members))
    # The stored blend must be what equal weights give, or every later number is off.
    stored = val[args.blend].to_numpy()
    if not np.allclose(ranks @ equal, stored, rtol=0, atol=1e-12):
        raise SystemExit(f"equal weights do not reproduce the stored {args.blend} column")

    drive_ids, y, dates = val["drive_id"].to_numpy(), val["label"].to_numpy(), val["date"]
    middle = dates.min() + (dates.max() - dates.min()) / 2
    in_a = (dates <= middle).to_numpy()
    full = DriveRows.build(drive_ids, y)
    ctx: dict = {
        "full": (full, full.arrange(ranks)),
        "halves": {},
        "budgets": tuple(args.far_budgets),
        "floors": tuple(args.recall_floors),
    }
    for name, mask in (("a", in_a), ("b", ~in_a)):
        rows = DriveRows.build(drive_ids[mask], y[mask])
        ctx["halves"][name] = (rows, rows.arrange(ranks[mask]))
    print(f"validation halves split at {middle}: {int(in_a.sum()):,} / {int((~in_a).sum()):,} rows")

    if args.weights:
        _score_given_weights(args, members, equal, full, ctx, criterion, summary_path)
        return

    grid = np.vstack([simplex_grid(len(members), args.step), equal])
    results = []
    for i, w in enumerate(grid):
        results.append(_evaluate(ctx, w, criterion))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(grid)} vectors, {time.perf_counter() - t0:.0f}s", flush=True)
    base = results[-1]  # equal weights, last
    check = {k: base[k] for k in ("pipeline_precision", "pipeline_recall")}
    print(f"equal weights: objective {base['objective']:.4f}, AP {base['average_precision']:.4f}, "
          f"pipeline point precision {check['pipeline_precision']:.4f} "
          f"recall {check['pipeline_recall']:.4f}")  # fmt: skip

    eligible = [
        r for r in results[:-1]
        if r["objective"] > base["objective"]
        and r["half_a"] >= base["half_a"]
        and r["half_b"] >= base["half_b"]
    ]  # fmt: skip
    verdict: dict = {"eligible": len(eligible), "searched": len(results)}
    recommended = None
    if eligible:
        best = max(eligible, key=lambda r: r["objective"])
        best_scores = full.drive_scores(ctx["full"][1] @ np.asarray(best["weights"]))
        base_scores = full.drive_scores(ctx["full"][1] @ equal)
        noise = bootstrap_difference(
            full.labels,
            best_scores,
            base_scores,
            n_boot=args.bootstrap,
            seed=args.seed,
            budgets=ctx["budgets"],
        )
        near = [r for r in eligible if r["objective"] >= best["objective"] - noise["se_a"]]
        recommended = min(
            near, key=lambda r: float(np.linalg.norm(np.asarray(r["weights"]) - equal))
        )
        rec_scores = full.drive_scores(ctx["full"][1] @ np.asarray(recommended["weights"]))
        gain = bootstrap_difference(
            full.labels,
            rec_scores,
            base_scores,
            n_boot=args.bootstrap,
            seed=args.seed + 1,
            budgets=ctx["budgets"],
        )
        verdict |= {
            "best_by_objective": best,
            "bootstrap_se_of_best": noise["se_a"],
            "near_best": len(near),
            "recommended": recommended,
            "gain_over_equal": gain,
            "reliable_improvement": bool(gain["ci_low"] > 0),
        }
    else:
        verdict["note"] = "no vector beats equal weights on full validation and on both halves"

    top = sorted(results, key=lambda r: -r["objective"])[:10]
    print("\ntop 10 by validation objective (weights: " + ", ".join(members) + "):")
    for r in top:
        print(f"  {r['weights']}  obj {r['objective']:.4f}  AP {r['average_precision']:.4f}  "
              f"halves {r['half_a']:.3f}/{r['half_b']:.3f}  "
              f"pipeline P {r['pipeline_precision']:.3f} R {r['pipeline_recall']:.3f}")  # fmt: skip
    if recommended:
        g = verdict["gain_over_equal"]
        print(f"\nrecommended {recommended['weights']}: objective gain over equal "
              f"{g['difference']:+.4f} [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}] "
              f"-> reliable improvement: {verdict['reliable_improvement']}")  # fmt: skip
    else:
        print(f"\n{verdict['note']}")

    test_rows = None
    if args.evaluate_test:
        if recommended is None:
            print("no recommended weights, so no test look was spent")
        else:
            test_rows = _evaluate_on_test(
                table, members, args.blend, recommended["weights"], config, reference
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "members": members,
        "blend": args.blend,
        "step": args.step,
        "far_budgets": list(ctx["budgets"]),
        "criterion": criterion,
        "equal": base,
        "all": results,
        "top": top,
        "verdict": verdict,
        "test": test_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {summary_path}")


def _score_given_weights(args, members, equal, full, ctx, criterion, summary_path) -> None:
    """Validation-only: each given weight vector against equal weights, with a paired bootstrap."""
    base = _evaluate(ctx, equal, criterion)
    base_scores = full.drive_scores(ctx["full"][1] @ equal)
    out = {"members": members, "equal": base, "given": []}
    print(f"members: {members}")
    for given in args.weights:
        w = np.asarray(given, dtype=np.float64)
        if len(w) != len(members) or (w < 0).any() or w.sum() <= 0:
            raise SystemExit(f"weights {given} do not fit members {members}")
        w = w / w.sum()
        row = _evaluate(ctx, w, criterion)
        gain = bootstrap_difference(
            full.labels,
            full.drive_scores(ctx["full"][1] @ w),
            base_scores,
            n_boot=args.bootstrap,
            seed=args.seed,
            budgets=ctx["budgets"],
        )
        out["given"].append({**row, "gain_over_equal": gain})
        far = [round(x, 3) for x in row["recall_at_far"]]
        far_equal = [round(x, 3) for x in base["recall_at_far"]]
        print(
            f"  {row['weights']}  objective {row['objective']:.4f} (equal {base['objective']:.4f})"
            f"  AP {row['average_precision']:.4f} (equal {base['average_precision']:.4f})\n"
            f"    date halves {row['half_a']:.3f}/{row['half_b']:.3f}"
            f" (equal {base['half_a']:.3f}/{base['half_b']:.3f})\n"
            f"    recall at false-alarm 1e-4/2e-4/5e-4: {far} vs {far_equal}\n"
            f"    precision at >= recall floor: {row['pipeline_precision']:.3f}"
            f" (recall {row['pipeline_recall']:.3f}) vs equal {base['pipeline_precision']:.3f}"
            f" (recall {base['pipeline_recall']:.3f})\n"
            f"    objective gain over equal {gain['difference']:+.4f}"
            f" [{gain['ci_low']:+.4f}, {gain['ci_high']:+.4f}]"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {summary_path}")


def _evaluate_on_test(table, members, blend, weights, config, reference) -> dict:
    """One recorded look at test: equal and recommended weights, same code path as train-merged."""
    from pipelines.train_merged_models import evaluate_model

    record_test_access(
        variant="blend_weight_tuning",
        purpose="recommended blend weights evaluated once on test",
        extra={"blend": blend, "members": members, "weights": weights},
    )
    splits = {name: table.filter(pl.col("split") == name) for name in ("validation", "test")}
    columns = {
        "equal": {name: frame[blend].to_numpy() for name, frame in splits.items()},
        "tuned": {
            name: rank_blend(reference, {m: frame[m].to_numpy() for m in members}, members, weights)
            for name, frame in splits.items()
        },
    }
    models = {}
    for label, by_split in columns.items():
        frames = {
            name: splits[name].select("drive_id", "label").with_columns(pl.Series(label, scores))
            for name, scores in by_split.items()
        }
        models[label] = evaluate_model(label, frames["validation"], frames["test"], config)

    lines = model_table({"models": models, "ensembles": {k: members for k in models}}).splitlines()
    names = {"equal": "equal weights", "tuned": f"weights {weights}"}
    for index, label in enumerate(models, start=2):
        lines[index] = re.sub(r"^\| [^|]* \|", f"| {names[label]} |", lines[index])
    print("\n" + "\n".join(lines))
    keep = ("validation_drive_level", "test_drive_level", "test_drive_ci", "threshold")
    return {label: {k: result[k] for k in keep} for label, result in models.items()}


if __name__ == "__main__":
    main()
