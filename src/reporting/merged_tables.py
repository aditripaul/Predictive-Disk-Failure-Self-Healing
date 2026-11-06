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


def _count(value: float) -> str:
    return f"{value:,.0f}"


def _status_row(label: str, caught: float, failing: float, false_alarms: float) -> str:
    """One `Precision | Recall | Caught | Missed | Missed % | False alarms` row."""
    flagged = caught + false_alarms
    recall = caught / failing if failing else 0.0
    return (
        f"| {label} | {_pct(caught / flagged if flagged else 0.0)} | {_pct(recall)} "
        f"| {_count(caught)} | {_count(failing - caught)} | {_pct(1 - recall)} "
        f"| {_count(false_alarms)} |"
    )


STATUS_HEADER = [
    "| Model | Precision | Recall | Caught | Missed | Missed % | False alarms |",
    "|---|---|---|---|---|---|---|",
]


def status_table(report: dict[str, Any], split: str = "test") -> str:
    """Runbook 1.1.3's status table for a `train-merged` report: each model and
    blend at its validation-chosen threshold, on `split` ("test" or "validation"),
    with the failing drives it missed."""
    ensembles = report.get("ensembles", {})
    lines = list(STATUS_HEADER)
    for name, result in report["models"].items():
        m = result[f"{split}_drive_level"]
        lines.append(
            _status_row(
                model_label(name, ensembles),
                m["caught_drive_count"],
                m["failing_drive_count"],
                m["false_alarm_drive_count"],
            )
        )
    return "\n".join(lines)


def recall_policy_table(summary: dict[str, Any], split: str = "test", k: str = "1") -> str:
    """Runbook 1.1.3's recall-first table from `scripts/run_recall_policy.py`'s
    `summary.json`: the warn net and each escalation tier, for hysteresis `k`.
    A catch there means warned at least `lead_min` days before the failure."""
    result = summary["results"][split][k]
    failing = result["failing_drives_judged"]
    lines = list(STATUS_HEADER)
    lines[0] = lines[0].replace("| Model |", "| Tier |")
    caught = result["caught_drives"]
    lines.append(_status_row("Warn", caught, failing, result["warned_drives"] - caught))
    for tier, entry in result["tiers"].items():
        lines.append(
            _status_row(
                tier.capitalize(),
                entry["caught"],
                failing,
                entry["escalated"] - entry["caught"],
            )
        )
    return "\n".join(lines)


def cascade_table(
    results: dict[str, Any],
    *,
    levels: tuple[float, ...] = (0.10, 0.60),
    production_net: float | None = None,
) -> str:
    """Runbook 1.1.3's cascade table from `make experiment-model --two-stage`'s
    `experiment_two_stage.json`: at each recall level, Stage 1 alone and the
    cascade of every net that reaches the level (means over seeds; test).

    A net narrower than the level cannot change it (its cascade equals Stage 1
    there), so those nets are named on the Stage 1 row instead."""
    kind_order = {"lightgbm": 0, "cox": 1, "aft": 2}
    nets = sorted(
        results["summary"],
        key=lambda n: (n["candidate_recall"], kind_order.get(n.get("stage2_kind", "lightgbm"), 9)),
    )
    lines = [
        "| Model | Recall level | Precision | Recall | Caught | Missed % | False alarms |",
        "|---|---|---|---|---|---|---|",
    ]

    def net_name(net: float, kind: str) -> str:
        name = f"{net * 100:.0f}% net"
        if kind != "lightgbm":
            name += f", {kind.upper() if kind == 'aft' else kind.capitalize()} Stage 2"
        return (
            f"{name} (production setting)" if production_net == net and kind == "lightgbm" else name
        )

    def row(label: str, level: float, entry: dict[str, Any], side: str) -> str:
        recall = entry[f"{side}_recall"]
        return (
            f"| {label} | {level * 100:.0f}% | {_pct(entry[f'{side}_precision'])} "
            f"| {_pct(recall)} | {_count(entry[f'{side}_caught_drive_count'])} "
            f"| {_pct(1 - recall)} | {_count(entry[f'{side}_false_alarm_drive_count'])} |"
        )

    for level in levels:
        reaching = [
            (net["candidate_recall"], net.get("stage2_kind", "lightgbm"), entry)
            for net in nets
            if net["candidate_recall"] > level + 1e-9
            for entry in net["levels"]
            if abs(entry["target_recall"] - level) < 1e-9
        ]
        if not reaching:
            continue
        narrower = sorted(
            {n["candidate_recall"] for n in nets if n["candidate_recall"] <= level + 1e-9}
        )
        stage1_label = "Stage 1 alone"
        if narrower:
            stage1_label += f" (= {', '.join(f'{n * 100:.0f}% net' for n in narrower)})"
        lines.append(row(stage1_label, level, reaching[0][2], "stage1"))
        for net, kind, entry in reaching:
            lines.append(row(f"Cascade, {net_name(net, kind)}", level, entry, "two_stage"))
    return "\n".join(lines)
