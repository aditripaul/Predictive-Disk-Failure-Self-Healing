"""Markdown tables for a `train-merged` run: the models, and the agent loop on each.

`model_table` reads a `merged_evaluation_report*.json` and prints every model and
blend as a row (validation precision; test precision with its bootstrap interval;
recall; drives caught and false alarms; AUROC and AUPRC). `agent_table` reads the
`pipelines/run_agent_fleet.py --score` report written per model by
`scripts/run_agent_models.py` and prints what the agent did with that model's scores.
"""

from __future__ import annotations

from typing import Any

#: Display names of the single models; blends are named from their members.
MODEL_NAMES = {
    "lightgbm": "LightGBM two-stage",
    "aft": "AFT",
    "cox": "Cox",
    "lstm": "LSTM",
    "tcn": "TCN",
}
MEMBER_NAMES = {"lightgbm": "LightGBM", "aft": "AFT", "cox": "Cox", "lstm": "LSTM", "tcn": "TCN"}
TIERS = ("warn", "cordon", "migrate", "drain")


def model_label(name: str, ensembles: dict[str, list[str]]) -> str:
    """`AFT`, or `Blend LightGBM + AFT + TCN` (members in their blend order)."""
    if name in MODEL_NAMES:
        return MODEL_NAMES[name]
    if name not in ensembles:
        raise KeyError(f"{name!r} is neither a model nor a blend in this report")
    return "Blend " + " + ".join(MEMBER_NAMES[m] for m in ensembles[name])


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def model_table(report: dict[str, Any]) -> str:
    """The per-model results table, in the report's model order."""
    ensembles = report.get("ensembles", {})
    lines = [
        "| Model | Val P | Test P [95% CI] | Test recall | Caught / false alarms | AUROC | AUPRC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, result in report["models"].items():
        val, test = result["validation_drive_level"], result["test_drive_level"]
        lo, hi = result["test_drive_ci"]["precision_ci"]
        lines.append(
            f"| {model_label(name, ensembles)} | {_pct(val['precision'])} "
            f"| {_pct(test['precision'])} [{lo * 100:.1f}, {hi * 100:.1f}] "
            f"| {_pct(test['recall'])} "
            f"| {test['caught_drive_count']} / {test['false_alarm_drive_count']} "
            f"| {test['auroc']:.3f} | {test['auprc']:.3f} |"
        )
    return "\n".join(lines)


def agent_table(
    runs: dict[str, dict[str, Any] | None],
    *,
    ensembles: dict[str, list[str]],
) -> str:
    """One row per model from its `run_agent_fleet --score` report (`None` = not run).

    `Acted` is the drives the agent cordoned, migrated or drained; precision and
    recall are the agent's, not the model's (`src/agent/backtest.py`). The run
    replays one snapshot (each drive's last row in the window), so these figures
    look ahead and are not point-in-time (runbook Section 4.7)."""
    lines = [
        "| Model | Candidates | Acted (with labels) | Caught / false actions "
        "| Agent precision | Agent recall | Warn / Cordon / Migrate / Drain | Human review |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        label = model_label(name, ensembles)
        if run is None:
            lines.append(f"| {label} | not run | | | | | | |")
            continue
        scoring = run.get("scoring") or {}
        if "agent_precision" not in scoring:
            lines.append(
                f"| {label} | {run['candidate_count']} | unscored: {scoring.get('skipped')} |||||"
            )
            continue
        labelled = scoring["drives_acted_on_with_labels"]
        tiers = scoring["actions_by_tier"]
        lines.append(
            f"| {label} | {run['candidate_count']} "
            f"| {scoring['drives_acted_on']} ({labelled}) "
            f"| {scoring['caught']} / {labelled - scoring['caught']} "
            f"| {_pct(scoring['agent_precision'])} | {_pct(scoring['agent_recall'])} "
            f"| {' / '.join(str(tiers.get(t, 0)) for t in TIERS)} "
            f"| {run['sent_to_human_review']} |"
        )
    return "\n".join(lines)
