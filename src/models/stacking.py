"""A stacked Stage 2: a validation-weighted rank blend of already-fitted Stage 2 models.

`make experiment-model --two-stage --stage2-kinds lightgbm cox stack` fits the LightGBM and
Cox Stage 2 of each net as before, then asks whether a blend of the two re-ranks the Stage 1
candidates better than either alone. Nothing is retrained: the members are the models already
fitted for that net.

The blend is the weighted mean of each member's rank against the VALIDATION candidate
distribution of the same member (`src/models/ensemble.py::validation_rank`), so test scores
are mapped without looking at test. The weights are searched on a simplex grid to maximise
drive-level average precision on the validation candidate rows. That is two or three numbers
fitted on thousands of rows, but the members also early-stopped on the same validation rows,
so the validation score of the blend is optimistic; only the paired test comparison counts.

Not logged to MLflow (`two_stage.log_two_stage` accepts a LightGBM Stage 2 only). A stack
used as a scorer is saved to and loaded from a folder instead (`save_stacked_two_stage`,
`load_stacked_two_stage`), as `scripts/freeze_stack_scorer.py` does for
`frozen_runs/stack_h60_seed0/`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from src.models.blend_weights import DriveRows, average_precision, simplex_grid
from src.models.ensemble import validation_rank
from src.models.training import predict_proba_positive
from src.models.two_stage import TwoStageModel, tree_count, with_stage1_score

#: The `stage2_kind` name of a stacked Stage 2 in the experiment.
STACK_KIND = "stack"


class StackedStage2:
    """Members' validation-ranked scores, blended. Same `predict_proba(x)[:, 1]`
    interface as every Stage 2, over the same input (features + Stage 1 score)."""

    def __init__(
        self,
        members: Mapping[str, Any],
        references: Mapping[str, np.ndarray],
        weights: Mapping[str, float],
    ) -> None:
        if set(members) != set(references) or set(members) != set(weights):
            raise ValueError("members, references and weights must name the same models")
        self.members = dict(members)
        self.references = {
            name: np.sort(np.asarray(r, dtype=np.float64)) for name, r in references.items()
        }
        self.weights = dict(weights)

    @property
    def tree_count(self) -> int:
        return sum(tree_count(m) for m in self.members.values())

    def blend(self, scores: Mapping[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(len(next(iter(scores.values()))), dtype=np.float64)
        for name, weight in self.weights.items():
            out += weight * validation_rank(self.references[name], scores[name])
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = {n: predict_proba_positive(m, x) for n, m in self.members.items()}
        blended = self.blend(scores)
        return np.column_stack([1.0 - blended, blended])


def fit_stack(
    stage1: Any,
    members: Mapping[str, Any],
    *,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_drives: np.ndarray,
    val_stage1_scores: np.ndarray,
    candidate_threshold: float,
    weight_step: float = 0.1,
) -> tuple[TwoStageModel | None, dict[str, Any]]:
    """Blends the fitted Stage 2 `members` ({name: model}) for one net.

    Returns the combined model and a report in the shape `fit_second_stage` gives, plus
    `stack`: the weights, and drive-level validation average precision of the blend, of
    equal weights and of each member alone. The model is None, with `report["status"]`
    saying why, when fewer than two members exist or validation has no failing candidate."""
    report: dict[str, Any] = {"candidate_threshold": float(candidate_threshold)}
    if len(members) < 2:
        report["status"] = "a stack needs at least two fitted Stage 2 models"
        return None, report
    val_rows = np.flatnonzero(val_stage1_scores >= candidate_threshold)
    if not len(val_rows) or y_val[val_rows].sum() == 0:
        report["status"] = "no failing validation candidates to fit stack weights on"
        return None, report

    names = list(members)
    x_stage2 = with_stage1_score(x_val, val_stage1_scores, val_rows)
    scores = {n: predict_proba_positive(members[n], x_stage2) for n in names}
    del x_stage2
    ranks = np.column_stack([validation_rank(scores[n], scores[n]) for n in names])

    rows = DriveRows.build(val_drives[val_rows], y_val[val_rows])
    arranged = rows.arrange(ranks)

    def drive_ap(w: np.ndarray) -> float:
        return average_precision(rows.labels, rows.drive_scores(arranged @ w))

    grid = simplex_grid(len(names), weight_step)
    values = np.array([drive_ap(w) for w in grid])
    best = grid[int(np.argmax(values))]
    equal = np.full(len(names), 1.0 / len(names))
    weights = {n: float(w) for n, w in zip(names, best, strict=True)}

    stack = StackedStage2(members, scores, weights)
    report["stack"] = {
        "members": names,
        "weights": weights,
        "weight_step": weight_step,
        "validation_drive_ap": {
            "stack": float(values.max()),
            "equal_weights": drive_ap(equal),
            **{n: drive_ap(np.eye(len(names))[i]) for i, n in enumerate(names)},
        },
    }
    report["stage2_trees"] = stack.tree_count
    report["status"] = "ok"
    return TwoStageModel(stage1, stack, candidate_threshold), report


#: Stage 2 member kinds a saved stack may hold, and how each is stored.
_LIGHTGBM_KIND, _COX_KIND = "lightgbm", "cox"


def _save_lightgbm(model: lgb.LGBMClassifier, directory: Path) -> None:
    import mlflow.lightgbm

    mlflow.lightgbm.save_model(model, str(directory))


def _load_lightgbm(directory: Path) -> Any:
    import mlflow.lightgbm

    return mlflow.lightgbm.load_model(str(directory))


def save_stacked_two_stage(model: TwoStageModel, out_dir: Path) -> dict[str, Any]:
    """Writes a two-stage model whose Stage 2 is a `StackedStage2` of LightGBM and Cox
    members to `out_dir`: `stage1/` and `stage2_lightgbm/` (MLflow LightGBM format, as
    the frozen runs store their stages), `stage2_cox/` (`CoxModel.save`),
    `references.npz` (each member's sorted validation candidate scores) and
    `stack.json` (weights, members, candidate threshold). Returns the `stack.json` dict.
    `out_dir` must not exist yet."""
    from src.models.survival import CoxModel

    stack = model.stage2
    if not isinstance(stack, StackedStage2):
        raise TypeError("save_stacked_two_stage needs a StackedStage2 Stage 2")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a frozen scorer is never overwritten")
    out_dir.mkdir(parents=True)
    _save_lightgbm(model.stage1, out_dir / "stage1")
    kinds: dict[str, str] = {}
    for name, member in stack.members.items():
        if isinstance(member, CoxModel):
            member.save(out_dir / f"stage2_{name}")
            kinds[name] = _COX_KIND
        elif isinstance(member, lgb.LGBMClassifier):
            _save_lightgbm(member, out_dir / f"stage2_{name}")
            kinds[name] = _LIGHTGBM_KIND
        else:
            raise TypeError(f"stack member {name!r} has an unsupported type {type(member)}")
    np.savez_compressed(out_dir / "references.npz", **stack.references)  # type: ignore[arg-type]
    meta = {
        "candidate_threshold": model.candidate_threshold,
        "weights": stack.weights,
        "members": kinds,
    }
    (out_dir / "stack.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_stacked_two_stage(out_dir: Path) -> TwoStageModel:
    """The `TwoStageModel` that `save_stacked_two_stage` wrote."""
    from src.models.survival import CoxModel

    meta = json.loads((out_dir / "stack.json").read_text())
    members: dict[str, Any] = {}
    for name, kind in meta["members"].items():
        directory = out_dir / f"stage2_{name}"
        members[name] = CoxModel.load(directory) if kind == _COX_KIND else _load_lightgbm(directory)
    with np.load(out_dir / "references.npz") as refs:
        references = {name: refs[name] for name in members}
    stack = StackedStage2(members, references, meta["weights"])
    return TwoStageModel(_load_lightgbm(out_dir / "stage1"), stack, meta["candidate_threshold"])
