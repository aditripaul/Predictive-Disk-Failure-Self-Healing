"""Per-family recalibration of the model score, shrunk toward the fleet.

The model's ranking quality varies 73-fold across drive families (runbook
4.9): the same `p_fail` carries a different real chance of failure depending
on the drive model. A single global threshold then cannot be optimal - it
flags too many drives of a family where a given score means less, and too few
where it means more. Recalibrating per family lets one threshold on the
recalibrated probability reorder flags ACROSS families. Within a family the
map is monotone, so it changes nothing about how drives of one model rank
against each other.

The form is deliberately small. A fleet-wide logistic recalibration on the
score's log-odds,

    P(fail) = sigmoid(a + b * logit(p_fail) + delta_family)

with only an intercept per family, and that intercept pulled toward zero by a
Gaussian prior of width `tau`. A family is often a few dozen failures; a
per-family slope, or per-family thresholds tuned to a precision target, fit
noise at that size - which is how per-family thresholds lost on test
(runbook 1.5). The intercept says one thing: in this family a given score
means k times the odds of failure. The prior keeps a family with five
failures near the fleet unless the evidence is strong.

`tau` is chosen without the test set: fit on the earlier half of the fitting
window's snapshot dates, score log-loss on the later half, keep the best, then
refit on the whole window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

EPS = 1e-6
DEFAULT_TAU_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clipped so an extreme log-odds cannot overflow exp().
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _log_likelihood(z: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(_sigmoid(z), EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_fleet(x: np.ndarray, y: np.ndarray, *, iterations: int = 100) -> tuple[float, float]:
    """Logistic regression of y on [1, logit(p_fail)] by damped Newton.

    Starts at the identity (a=0, b=1): the score is already roughly a
    probability, so that is close. Starting the intercept at the base rate's
    log-odds double-counts the base rate - the score's log-odds already carry
    it - and the first undamped step overshoots so far that the next Hessian
    is singular. Each step is halved until the likelihood stops falling, so a
    poor start or a near-separable family cannot make it diverge."""
    a, b = 0.0, 1.0
    current = _log_likelihood(a + b * x, y)
    for _ in range(iterations):
        p = _sigmoid(a + b * x)
        w = p * (1 - p)
        g = np.array([np.sum(y - p), np.sum((y - p) * x)])
        h = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]])
        step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        scale = 1.0
        while scale > 1e-6:
            na, nb = a + scale * step[0], b + scale * step[1]
            candidate = _log_likelihood(na + nb * x, y)
            if candidate >= current - 1e-12:
                break
            scale /= 2
        a, b, current = na, nb, candidate
        if np.max(np.abs(scale * step)) < 1e-9:
            break
    return float(a), float(b)


def _fit_offsets(
    eta: np.ndarray, y: np.ndarray, codes: np.ndarray, n: int, tau: float, *, iterations: int = 50
) -> np.ndarray:
    """MAP intercept per family under delta ~ N(0, tau^2), all families at
    once: each family's Newton step depends only on its own rows."""
    delta = np.zeros(n)
    prior = 1.0 / (tau * tau)
    for _ in range(iterations):
        p = _sigmoid(eta + delta[codes])
        g = np.bincount(codes, weights=y - p, minlength=n) - prior * delta
        h = np.bincount(codes, weights=p * (1 - p), minlength=n) + prior
        step = g / h
        delta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return delta


@dataclass
class FamilyCalibrator:
    """Fleet logistic recalibration plus a shrunk intercept per family."""

    family_column: str = "drive_model"
    tau: float = 0.5
    a: float = 0.0
    b: float = 1.0
    offsets: dict[str, float] = field(default_factory=dict)
    failures_by_family: dict[str, int] = field(default_factory=dict)

    def fit(self, obs: pl.DataFrame) -> FamilyCalibrator:
        x = _logit(obs["p_fail"].to_numpy())
        y = (obs["label"].to_numpy() == 1).astype(float)
        self.a, self.b = _fit_fleet(x, y)
        families = obs[self.family_column].fill_null("(unknown)").to_numpy()
        names, codes = np.unique(families, return_inverse=True)
        delta = _fit_offsets(self.a + self.b * x, y, codes, len(names), self.tau)
        self.offsets = {str(n): float(d) for n, d in zip(names, delta, strict=True)}
        failures = np.bincount(codes, weights=y, minlength=len(names))
        self.failures_by_family = {str(n): int(f) for n, f in zip(names, failures, strict=True)}
        return self

    def transform(self, obs: pl.DataFrame) -> np.ndarray:
        """Recalibrated P(fail). A family never seen while fitting gets offset
        0 - the fleet calibration - rather than an error."""
        x = _logit(obs["p_fail"].to_numpy())
        families = obs[self.family_column].fill_null("(unknown)").to_list()
        delta = np.array([self.offsets.get(f, 0.0) for f in families])
        return _sigmoid(self.a + self.b * x + delta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_column": self.family_column,
            "tau": self.tau,
            "a": self.a,
            "b": self.b,
            "offsets": self.offsets,
            "failures_by_family": self.failures_by_family,
        }


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select_tau(
    obs: pl.DataFrame,
    *,
    family_column: str = "drive_model",
    grid: tuple[float, ...] = DEFAULT_TAU_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """The prior width that best predicts LATER snapshots from EARLIER ones,
    within the fitting window only. Splitting by date rather than at random
    keeps one drive's near-identical weekly rows from sitting on both sides,
    which would reward overfitting. `tau=0` is the fleet-only baseline and is
    always included, so "no family effect" can win."""
    dates = sorted(obs["date"].unique().to_list())
    if len(dates) < 2:
        raise ValueError("need at least two snapshot dates to choose tau")
    cut = dates[len(dates) // 2]
    early, late = obs.filter(pl.col("date") < cut), obs.filter(pl.col("date") >= cut)
    y_late = (late["label"].to_numpy() == 1).astype(float)

    fleet = FamilyCalibrator(family_column=family_column, tau=1e-6).fit(early)
    scores = [{"tau": 0.0, "log_loss": log_loss(y_late, fleet.transform(late))}]
    for tau in grid:
        cal = FamilyCalibrator(family_column=family_column, tau=tau).fit(early)
        scores.append({"tau": tau, "log_loss": log_loss(y_late, cal.transform(late))})
    best = min(scores, key=lambda r: r["log_loss"])
    return best["tau"], scores
