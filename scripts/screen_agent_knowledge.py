"""The pre-registered screen of knowledge rules (docs/agent_knowledge_base.md section 8,
with Amendment 1). Validation only by default; the Q3 step is guarded.

    uv run python scripts/screen_agent_knowledge.py            # validation screen
    uv run python scripts/screen_agent_knowledge.py --q3       # one Q3 look, only after a pass

Screen set: daily validation observations of the frozen stack scorer, 2026-02-01..03-02
(every 60-day horizon ends by 2026-05-01), each flagged drive joined to its knowledge
context as of that day. Select half 02-01..02-15 chooses each rule's hyperparameter
(recall at the judged tier's own false-alarm rate, ties to "no rule"); confirm half
02-16..03-02 judges it under criteria 1-4. No test row is read and nothing is logged in
the test-access log unless `--q3` runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.family_calibration import FamilyCalibrator, select_tau  # noqa: E402
from src.agent.knowledge.rules import GRIDS, RULES, apply_tier_rule, tier_of  # noqa: E402
from src.agent.knowledge.store import KnowledgeBase  # noqa: E402
from src.agent.point_in_time import compare_scorings, dense_drive_codes  # noqa: E402
from src.logging_config import configure_logging, get_logger  # noqa: E402
from src.models.threshold import resolve_action_thresholds  # noqa: E402
from src.resource_limits import apply_memory_limit_from_config  # noqa: E402

logger = get_logger(__name__)
DOC = Path("docs/agent_knowledge_base.md")
SCREEN_FILES = (
    DOC,
    Path("scripts/screen_agent_knowledge.py"),
    Path("src/agent/knowledge/rules.py"),
)
BUDGETS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3)
JUDGED_TIERS = ("warn", "cordon")
MIN_CHANGED_DAYS, MIN_CHANGED_DRIVES = 300, 60
R1_TAU_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)  # tau = 0 is always added by select_tau


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


# --- the screen set -----------------------------------------------------------------


def screen_set(
    kb: KnowledgeBase,
    scores_path: Path,
    *,
    split: str,
    column: str,
    start: dt.date,
    end: dt.date,
    warn: float,
) -> tuple[pl.DataFrame, dict[tuple[str, dt.date], dict[str, Any] | None]]:
    """Observations `drive_id, date, p_fail, label, drive_model` and the knowledge context
    of every observation at or above warn (rules leave the others unchanged)."""
    frames: list[pl.DataFrame] = []
    contexts: dict[tuple[str, dt.date], dict[str, Any] | None] = {}
    models = dict(kb.db.execute("SELECT drive_id, drive_model FROM drive_info").fetchall())
    day = start
    while day <= end:
        obs = (
            pl.scan_parquet(scores_path)
            .filter((pl.col("date") == day) & (pl.col("split") == split))
            .select("drive_id", "date", pl.col(column).alias("p_fail"), "label")
            .collect()
        )
        flagged = obs.filter(pl.col("p_fail") >= warn)["drive_id"].to_list()
        for drive, ctx in kb.contexts(flagged, as_of=day).items():
            contexts[(drive, day)] = ctx.to_json() if ctx.known else None
        frames.append(obs)
        day += dt.timedelta(days=1)
    out = pl.concat(frames)
    return out.with_columns(
        pl.col("drive_id").replace_strict(models, default=None).alias("drive_model")
    ), contexts


def context_list(obs: pl.DataFrame, contexts: dict) -> list[dict | None]:
    return [
        contexts.get((d, day))
        for d, day in zip(obs["drive_id"].to_list(), obs["date"].to_list(), strict=True)
    ]


# --- metrics -------------------------------------------------------------------------


def tier_flags(p: np.ndarray, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    return {t: p >= thresholds[t] for t in JUDGED_TIERS}


def far_of(flags: np.ndarray, failed: np.ndarray) -> float:
    negatives = (~failed).sum()
    return float((flags & ~failed).sum() / negatives) if negatives else 0.0


def paired_difference(
    codes: np.ndarray,
    failed: np.ndarray,
    flags_a: np.ndarray,
    flags_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Precision and recall of B minus A at fixed flags, with a paired bootstrap that
    resamples whole drives (both arms on the same resamples)."""
    n = int(codes.max()) + 1
    per = {
        key: np.bincount(codes, weights=v.astype(float), minlength=n)
        for key, v in {
            "fa": flags_a,
            "ca": flags_a & failed,
            "fb": flags_b,
            "cb": flags_b & failed,
            "pos": failed,
        }.items()
    }

    def stats(w: np.ndarray) -> tuple[float, float, float, float]:
        fa, ca, fb, cb, pos = (float(per[k] @ w) for k in ("fa", "ca", "fb", "cb", "pos"))
        pa, pb = (ca / fa if fa else 0.0), (cb / fb if fb else 0.0)
        ra, rb = (ca / pos if pos else 0.0), (cb / pos if pos else 0.0)
        return pa, pb, ra, rb

    pa, pb, ra, rb = stats(np.ones(n))
    rng = np.random.default_rng(seed)
    dp, dr = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        a_p, b_p, a_r, b_r = stats(w)
        dp[i], dr[i] = b_p - a_p, b_r - a_r
    return {
        "precision_a": pa,
        "precision_b": pb,
        "recall_a": ra,
        "recall_b": rb,
        "precision_difference": pb - pa,
        "precision_ci": [float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))],
        "recall_difference": rb - ra,
        "recall_ci": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
    }


def recall_at_rate(p: np.ndarray, failed: np.ndarray, rate: float) -> float:
    """Recall of the most permissive threshold whose false-alarm rate is at most `rate`."""
    order = np.argsort(-p, kind="stable")
    fa = np.cumsum(~failed[order]) / max(1, (~failed).sum())
    caught = np.cumsum(failed[order]) / max(1, failed.sum())
    ok = np.flatnonzero(fa <= rate)
    return float(caught[ok[-1]]) if len(ok) else 0.0


def matched_recall_threshold(p: np.ndarray, failed: np.ndarray, recall: float) -> float:
    """The highest threshold on `p` whose recall reaches `recall`."""
    positives = np.sort(p[failed])[::-1]
    need = max(1, int(np.ceil(recall * len(positives))))
    return float(positives[min(need, len(positives)) - 1]) if len(positives) else 1.01


# --- one rule ------------------------------------------------------------------------


def candidate_scores(
    family: str, params: dict, obs: pl.DataFrame, ctx: list, thresholds: dict, fitted: dict
) -> tuple[np.ndarray, dict[str, float]]:
    """Transformed scores and the thresholds to judge them at. Tier rules keep the
    scorer's thresholds; R1 changes the scale, so its thresholds are set to flag the same
    number of select-half drive-days as the baseline does at each tier."""
    p = obs["p_fail"].to_numpy()
    if family == "none":
        return p, thresholds
    if family == "r1_cohort_prior":
        cal = fitted.get(("r1", params["tau"]))
        q = cal.transform(obs) if cal is not None else p
        return q, fitted[("r1_thresholds", params["tau"])]
    return apply_tier_rule(RULES[family], p, ctx, params, thresholds), thresholds


def evaluate(
    obs: pl.DataFrame,
    p_b: np.ndarray,
    thr_b: dict,
    thresholds: dict,
    *,
    judged: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Criteria 1-3 for one candidate scoring against "no rule" on `obs`."""
    failed = obs["label"].to_numpy() == 1
    codes = dense_drive_codes(obs)
    p_a = obs["p_fail"].to_numpy()
    flags_a, flags_b = tier_flags(p_a, thresholds), tier_flags(p_b, thr_b)
    out: dict[str, Any] = {"tiers": {}}
    c1 = c2 = True
    for tier in judged:
        diff = paired_difference(
            codes, failed, flags_a[tier], flags_b[tier], n_boot=n_boot, seed=seed
        )
        c1_tier = diff["precision_ci"][0] > 0 and diff["recall_ci"][1] >= 0
        rate = far_of(flags_a[tier], failed)
        below = [b for b in BUDGETS if b <= rate]
        above = [b for b in BUDGETS if b >= rate]
        bracket = tuple(
            sorted({below[-1] if below else BUDGETS[0], above[0] if above else BUDGETS[-1]})
        )
        far = compare_scorings(
            obs["drive_id"], failed, p_a, p_b, budgets=bracket, n_boot=min(n_boot, 500), seed=seed
        )
        far_ok = all(r["difference"] > 0 and r["difference_ci"][0] >= 0 for r in far)
        plain_thr = matched_recall_threshold(p_a, failed, diff["recall_b"])
        plain = paired_difference(
            codes, failed, p_a >= plain_thr, flags_b[tier], n_boot=n_boot, seed=seed
        )
        plain_ok = plain["precision_ci"][0] > 0
        c1, c2 = c1 and c1_tier, c2 and far_ok and plain_ok
        out["tiers"][tier] = {
            "difference": diff,
            "criterion_1": c1_tier,
            "baseline_false_alarm_rate": rate,
            "far_budgets": list(bracket),
            "recall_at_far": far,
            "plain_threshold_at_matched_recall": plain_thr,
            "against_plain_threshold": plain,
            "criterion_2": far_ok and plain_ok,
        }
    flagged_any = (p_a >= thresholds["warn"]) | (p_b >= thr_b["warn"])
    tier_a = np.array([tier_of(x, thresholds).value for x in p_a[flagged_any]])
    tier_b = np.array([tier_of(x, thr_b).value for x in p_b[flagged_any]])
    changed = tier_a != tier_b
    drives_changed = obs["drive_id"].to_numpy()[flagged_any][changed]
    out["changed_drive_days"] = int(changed.sum())
    out["changed_drives"] = len(set(drives_changed.tolist()))
    out["criterion_1"], out["criterion_2"] = c1, c2
    out["criterion_3"] = (
        out["changed_drive_days"] >= MIN_CHANGED_DAYS
        and out["changed_drives"] >= MIN_CHANGED_DRIVES
    )
    return out


def judged_tiers(
    obs: pl.DataFrame, p_b: np.ndarray, thr_b: dict, thresholds: dict
) -> tuple[str, ...]:
    """Amendment 1: a rule whose warn-and-above flags equal "no rule"'s is judged at
    cordon-and-above alone; otherwise at both."""
    same_warn = np.array_equal(obs["p_fail"].to_numpy() >= thresholds["warn"], p_b >= thr_b["warn"])
    return ("cordon",) if same_warn else JUDGED_TIERS


def select_param(
    family: str, grid: list[dict], sel: pl.DataFrame, ctx: list, thresholds: dict, fitted: dict
) -> dict:
    """Recall at the judged tiers' own false-alarm rates on the select half, mean over the
    judged tiers; the best value must beat "no rule" strictly, else "no rule"."""
    failed = sel["label"].to_numpy() == 1
    p_a = sel["p_fail"].to_numpy()
    rows: list[dict[str, Any]] = []
    for params in grid:
        p_b, thr_b = candidate_scores(family, params, sel, ctx, thresholds, fitted)
        tiers = judged_tiers(sel, p_b, thr_b, thresholds)
        score = base = 0.0
        for tier in tiers:
            rate = far_of(p_a >= thresholds[tier], failed)
            score += recall_at_rate(p_b, failed, rate) / len(tiers)
            base += recall_at_rate(p_a, failed, rate) / len(tiers)
        rows.append(
            {"params": params, "judged": list(tiers), "recall": score, "baseline_recall": base}
        )
    best = max(rows, key=lambda r: r["recall"] - r["baseline_recall"])
    chosen = best["params"] if best["recall"] > best["baseline_recall"] + 1e-12 else None
    return {"grid": rows, "chosen": chosen}


def fit_r1(sel: pl.DataFrame, thresholds: dict) -> tuple[dict, list[dict]]:
    """R1: per tau, a calibrator fitted on the select half, and thresholds that flag the
    same number of select-half drive-days as the baseline at each tier."""
    fitted: dict = {}
    tau_choice, tau_scores = select_tau(sel, grid=R1_TAU_GRID)
    grid = [{"tau": 0.0}] + [{"tau": t} for t in R1_TAU_GRID]
    p_a = sel["p_fail"].to_numpy()
    for params in grid:
        cal = FamilyCalibrator(tau=params["tau"] or 1e-6).fit(sel)
        q = cal.transform(sel)
        fitted[("r1", params["tau"])] = cal
        order = np.sort(q)[::-1]
        thr = {}
        for tier in ("warn", "cordon", "migrate", "drain"):
            k = int((p_a >= thresholds[tier]).sum())
            thr[tier] = float(order[k - 1]) if k > 0 else 1.01
        fitted[("r1_thresholds", params["tau"])] = thr
    fitted["tau_selection"] = {"chosen_by_log_loss": tau_choice, "scores": tau_scores}
    return fitted, grid


# --- main ----------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def q3_guard(out_dir: Path) -> list[str]:
    """Reasons the Q3 step must not run; empty means it may."""
    reasons = []
    if not _git("log", "-1", "--format=%H", "--", str(DOC)):
        reasons.append(f"{DOC} is not committed")
    if _git("status", "--porcelain", "--", *map(str, SCREEN_FILES)):
        reasons.append("the design doc, the screen or the rules have uncommitted changes")
    validation = out_dir / "validation.json"
    if not validation.exists():
        reasons.append("no validation.json: run the validation screen first")
    elif not any(
        r.get("rule_passed") for r in json.loads(validation.read_text())["rules"].values()
    ):
        reasons.append("no rule passed the validation screen: there is nothing to look at on Q3")
    if (out_dir / "q3_evaluation.json").exists():
        reasons.append("q3_evaluation.json exists: the one Q3 look has been taken")
    return reasons


def tables_markdown(report: dict) -> str:
    lines = [
        "# Knowledge-rule screen (validation)",
        "",
        f"Screen set {report['screen']['start']}..{report['screen']['end']}, select half to "
        f"{report['screen']['select_end']}, confirm half from {report['screen']['confirm_start']}; "
        f"scorer `{report['scorer']}`, tier set `{report['tier_set']}`. "
        f"Commit `{report['git_commit']}`.",
        "",
        f"Confirm half: {report['confirm']['observations']:,} drive-days, "
        f"{report['confirm']['drives']:,} "
        f"drives, row failure rate {report['confirm']['row_failure_rate']:.3%}.",
        "",
        "| Rule | Chosen | Judged at | Changed drive-days (drives) | Precision diff [95% CI] | "
        "Recall diff [95% CI] | C1 | C2 | C3 | C4 | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in report["rules"].items():
        if r.get("status") == "not_screenable":
            lines.append(
                f"| {name} | - | - | - | - | - | - | - | - | - | not screenable: {r['reason']} |"
            )
            continue
        ev = r.get("confirm")
        if ev is None:
            lines.append(f"| {name} | none | - | - | - | - | - | - | - | no | {r['verdict']} |")
            continue
        cells = []
        for tier in ev["tiers"]:
            d = ev["tiers"][tier]["difference"]
            cells.append(
                (
                    f"{tier}: {100 * d['precision_difference']:+.1f} "
                    f"[{100 * d['precision_ci'][0]:+.1f}, "
                    f"{100 * d['precision_ci'][1]:+.1f}]",
                    f"{tier}: {100 * d['recall_difference']:+.1f} [{100 * d['recall_ci'][0]:+.1f}, "
                    f"{100 * d['recall_ci'][1]:+.1f}]",
                )
            )
        lines.append(
            f"| {name} | {json.dumps(r['chosen'])} | {', '.join(ev['tiers'])} | "
            f"{ev['changed_drive_days']:,} ({ev['changed_drives']:,}) | "
            f"{'; '.join(c[0] for c in cells)} | "
            f"{'; '.join(c[1] for c in cells)} | {'yes' if ev['criterion_1'] else 'no'} | "
            f"{'yes' if ev['criterion_2'] else 'no'} | {'yes' if ev['criterion_3'] else 'no'} | "
            f"{'yes' if r['criterion_4'] else 'no'} | {r['verdict']} |"
        )
    lines += [
        "",
        "C1-C4 are the criteria of docs/agent_knowledge_base.md section 8 (Amendment 1 applied);",
        'points are precision or recall points, rule minus "no rule", per drive-day.',
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--knowledge", type=Path, default=Path("data/audit/agent_knowledge.sqlite"))
    parser.add_argument(
        "--scores", type=Path, default=Path("data/tmp/stack_scores/scores_h60_seed0.parquet")
    )
    parser.add_argument("--score-column", default="score")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("frozen_runs/stack_h60_seed0/agent_thresholds_stack_h60_seed0.json"),
    )
    parser.add_argument("--tier-key", default="action_tiers")
    parser.add_argument("--start", type=_date, default=_date("2026-02-01"))
    parser.add_argument("--select-end", type=_date, default=_date("2026-02-15"))
    parser.add_argument("--end", type=_date, default=_date("2026-03-02"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/audit/agent_knowledge_h60"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q3", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()
    apply_memory_limit_from_config()

    if args.q3:
        reasons = q3_guard(args.out_dir)
        if reasons:
            raise SystemExit("the Q3 step refuses to run:\n  - " + "\n  - ".join(reasons))
        raise SystemExit(
            "a rule passed: the Q3 look needs the stack's Q3 scores and the knowledge store "
            "advanced through Q3, which this script does not build yet "
            "(docs/agent_knowledge_base.md section 9)"
        )

    meta = json.loads(args.thresholds.read_text())
    thresholds = resolve_action_thresholds(meta[args.tier_key], {})
    kb = KnowledgeBase(args.knowledge)
    obs, contexts = screen_set(
        kb,
        args.scores,
        split="validation",
        column=args.score_column,
        start=args.start,
        end=args.end,
        warn=thresholds["warn"],
    )
    sel = obs.filter(pl.col("date") <= args.select_end)
    conf = obs.filter(pl.col("date") > args.select_end)
    ctx_sel, ctx_conf = context_list(sel, contexts), context_list(conf, contexts)
    r1_fitted, r1_grid = fit_r1(sel, thresholds)

    report: dict[str, Any] = {
        "screen": {
            "start": str(args.start),
            "select_end": str(args.select_end),
            "confirm_start": str(args.select_end + dt.timedelta(days=1)),
            "end": str(args.end),
        },
        "scorer": str(args.thresholds),
        "tier_set": args.tier_key,
        "thresholds": thresholds,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "select": {"observations": sel.height, "drives": sel["drive_id"].n_unique()},
        "confirm": {
            "observations": conf.height,
            "drives": conf["drive_id"].n_unique(),
            "row_failure_rate": float((conf["label"] == 1).sum()) / max(1, conf.height),
            "with_context": sum(1 for c in ctx_conf if c is not None),
        },
        "r1_tau_selection": r1_fitted["tau_selection"],
        "rules": {},
    }
    families = {
        "r1_cohort_prior": r1_grid,
        **{k: v for k, v in GRIDS.items() if k != "r3_outcome_memory"},
    }
    for family, grid in families.items():
        selection = select_param(family, grid, sel, ctx_sel, thresholds, r1_fitted)
        entry: dict[str, Any] = {"selection": selection, "chosen": selection["chosen"]}
        if selection["chosen"] is None:
            entry.update(
                rule_passed=False,
                criterion_4=False,
                verdict='failed: no value beat "no rule" on the select half',
            )
            report["rules"][family] = entry
            continue
        p_b, thr_b = candidate_scores(
            family, selection["chosen"], conf, ctx_conf, thresholds, r1_fitted
        )
        judged = judged_tiers(conf, p_b, thr_b, thresholds)
        ev = evaluate(
            conf, p_b, thr_b, thresholds, judged=judged, n_boot=args.bootstrap, seed=args.seed
        )
        c4 = True
        values = [list(g.values())[0] for g in grid]
        chosen_value = list(selection["chosen"].values())[0]
        if family == "r2_chronic_demote" and chosen_value == values[-1]:
            inner = {"n_days": values[-2]}
            p_i, thr_i = candidate_scores(family, inner, conf, ctx_conf, thresholds, r1_fitted)
            inner_ev = evaluate(
                conf, p_i, thr_i, thresholds, judged=judged, n_boot=args.bootstrap, seed=args.seed
            )
            c4 = inner_ev["criterion_1"]
            entry["edge_check"] = {"params": inner, "criterion_1": inner_ev["criterion_1"]}
        passed = ev["criterion_1"] and ev["criterion_2"] and ev["criterion_3"] and c4
        if passed:
            verdict = "passed"
        elif not ev["criterion_3"]:
            verdict = "underpowered"
        else:
            verdict = "failed"
        entry.update(confirm=ev, criterion_4=c4, rule_passed=passed, verdict=verdict)
        report["rules"][family] = entry
    report["rules"]["r3_outcome_memory"] = {
        "status": "not_screenable",
        "reason": (
            "no validation date has an episode resolved 60 days after it closed (section 7.1)"
        ),
        "rule_passed": False,
        "verdict": "not screenable",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "validation.json").write_text(json.dumps(report, indent=2, default=str))
    (args.out_dir / "tables.md").write_text(tables_markdown(report))
    for name, r in report["rules"].items():
        print(f"{name:<24} {r['verdict']}")
    print(f"written to {args.out_dir / 'validation.json'} and tables.md")


if __name__ == "__main__":
    main()
