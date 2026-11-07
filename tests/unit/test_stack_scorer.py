"""Saving and loading the stacked Stage 2 and its Cox member (frozen scorer, Phase S)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from src.models.ensemble import validation_rank
from src.models.stacking import StackedStage2, load_stacked_two_stage, save_stacked_two_stage
from src.models.survival import CoxModel, train_cox
from src.models.two_stage import TwoStageModel, with_stage1_score


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    y = (x[:, 0] + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    return x, y


def _models(x, y):
    stage1 = lgb.LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1, random_state=0).fit(x, y)
    s1 = stage1.predict_proba(x)[:, 1]
    rows = np.arange(len(y))
    x2 = with_stage1_score(x, s1, rows)
    lgbm2 = lgb.LGBMClassifier(n_estimators=15, num_leaves=5, verbose=-1, random_state=0).fit(x2, y)
    time = np.where(y == 1, 10.0, 60.0)
    lower, upper = time, np.where(y == 1, time, np.inf)
    cox = train_cox(
        x2, lower, upper, eval_x=x2, eval_y=y, num_boost_round=30, early_stopping_rounds=5
    )
    return stage1, s1, x2, lgbm2, cox


def test_cox_reload_predicts_identically(tmp_path):
    x, y = _data()
    _, _, x2, _, cox = _models(x, y)
    cox.save(tmp_path / "cox")
    again = CoxModel.load(tmp_path / "cox")
    assert again.tree_count == cox.tree_count
    np.testing.assert_array_equal(again.predict_proba(x2), cox.predict_proba(x2))


def test_stack_round_trip_gives_the_same_final_scores(tmp_path):
    x, y = _data()
    stage1, s1, x2, lgbm2, cox = _models(x, y)
    members = {"lightgbm": lgbm2, "cox": cox}
    references = {n: m.predict_proba(x2)[:, 1] for n, m in members.items()}
    model = TwoStageModel(
        stage1, StackedStage2(members, references, {"lightgbm": 0.5, "cox": 0.5}), 0.3
    )
    meta = save_stacked_two_stage(model, tmp_path / "stack")
    assert meta["members"] == {"lightgbm": "lightgbm", "cox": "cox"}
    loaded = load_stacked_two_stage(tmp_path / "stack")
    assert loaded.candidate_threshold == pytest.approx(0.3)
    np.testing.assert_allclose(
        loaded.final_scores(x, s1), model.final_scores(x, s1), rtol=0, atol=1e-12
    )
    # rejects keep their Stage 1 score; candidates sit on [threshold, 1]
    final = loaded.final_scores(x, s1)
    assert np.array_equal(final[s1 < 0.3], s1[s1 < 0.3])
    assert ((final[s1 >= 0.3] >= 0.3) & (final[s1 >= 0.3] <= 1.0)).all()
    # the blend is the weighted mean of validation ranks
    rows = s1 >= 0.3
    blended = 0.5 * validation_rank(references["lightgbm"], lgbm2.predict_proba(x2[rows])[:, 1])
    blended += 0.5 * validation_rank(references["cox"], cox.predict_proba(x2[rows])[:, 1])
    np.testing.assert_allclose(final[rows], 0.3 + 0.7 * blended, atol=1e-12)


def test_a_frozen_scorer_is_never_overwritten(tmp_path):
    x, y = _data()
    stage1, s1, x2, lgbm2, cox = _models(x, y)
    members = {"lightgbm": lgbm2, "cox": cox}
    refs = {n: m.predict_proba(x2)[:, 1] for n, m in members.items()}
    model = TwoStageModel(stage1, StackedStage2(members, refs, {"lightgbm": 1.0, "cox": 0.0}), 0.3)
    (tmp_path / "stack").mkdir()
    with pytest.raises(FileExistsError):
        save_stacked_two_stage(model, tmp_path / "stack")
    with pytest.raises(TypeError):
        save_stacked_two_stage(TwoStageModel(stage1, lgbm2, 0.3), tmp_path / "other")
