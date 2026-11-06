"""Failure-mode clusters of suspect drives (docs/cluster_action_policy.md).

A drive's *score* says how likely it is to fail; its *cluster* says what kind of
problem its SMART signature shows (media wearing out, pending sectors piling up,
interface errors, ...). The cluster is built only from signature columns of the
gold feature table on the decision day - never the score, a label or the drive
vendor - so it is a second axis beside the score tier, not a re-ranking of it.

`ClusterModel` holds everything needed to assign a new drive: the column list,
the robust scaler, and either centroids (nearest-centroid assignment) or the
parameters of a Gaussian mixture (highest posterior). Assignment is plain numpy,
so the agent can run it without scikit-learn.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

CORE = (
    "reallocated_sector_count",
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",
    "command_timeout",
    "reallocation_event_count",
)
#: Non-negative counts: log1p.
COUNT_COLUMNS = (*CORE, "active_defect_total", "udma_crc_error_count")
#: Signed rates of change: sign * log1p(|x|).
SIGNED_COLUMNS = (
    *(f"{c}_30d_delta" for c in CORE),
    *(f"{c}_30d_slope" for c in CORE),
    "active_defect_velocity_7d",
    "active_defect_velocity_30d",
    "active_defect_acceleration_7d_vs_30d",
    "udma_crc_error_count_30d_delta",
    "seek_error_rate_30d_delta",
    "temperature_spike",
)
#: Days since a counter last rose; 999 means never. Capped at 365, then log1p.
DAYS_COLUMNS = tuple(f"{c}_days_since_last_increase" for c in CORE)
#: Booleans: 0 / 1.
FLAG_COLUMNS = tuple(f"{c}_zero_to_nonzero" for c in CORE)
PLAIN_COLUMNS = ("temperature_celsius", "telemetry_coverage_30d")
#: Gold columns the signature is built from.
SOURCE_COLUMNS = (
    *COUNT_COLUMNS,
    *SIGNED_COLUMNS,
    *DAYS_COLUMNS,
    *FLAG_COLUMNS,
    *PLAIN_COLUMNS,
    "pending_to_reallocated_ratio",
    "power_on_hours",
)
DAYS_CAP = 365

#: Readable names for the signature columns, used to name clusters.
READABLE = {
    "reallocated_sector_count": "reallocated sectors",
    "current_pending_sector_count": "pending sectors",
    "offline_uncorrectable": "offline uncorrectable",
    "reported_uncorrectable_errors": "reported uncorrectable",
    "command_timeout": "command timeouts",
    "reallocation_event_count": "reallocation events",
    "active_defect_total": "active defects",
    "udma_crc_error_count": "CRC errors",
    "power_on_years": "old drive",
    "temperature_celsius": "hot",
    "telemetry_coverage_30d": "full telemetry",
    "pending_to_reallocated_ratio": "pending vs reallocated",
}
#: Features whose cluster means a cable / backplane problem more than the disk.
INTERFACE_FEATURES = ("udma_crc_error_count", "command_timeout")
#: Features whose cluster means data at risk: move it before it is unreadable.
MEDIA_AT_RISK_FEATURES = (
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",
)


def _signed_log(col: str) -> pl.Expr:
    x = pl.col(col).cast(pl.Float64).fill_null(0.0)
    return (x.sign() * x.abs().log1p()).alias(col)


def signature_frame(rows: pl.DataFrame) -> pl.DataFrame:
    """The transformed signature of each row, plus `drive_id` and `date` when
    present. Missing values count as "nothing happened" (0, or never for the
    days-since columns)."""
    keep = [c for c in ("drive_id", "date") if c in rows.columns]
    exprs: list[pl.Expr] = [pl.col(c) for c in keep]
    exprs += [
        pl.col(c).cast(pl.Float64).fill_null(0.0).clip(0.0).log1p().alias(c) for c in COUNT_COLUMNS
    ]
    exprs += [_signed_log(c) for c in SIGNED_COLUMNS]
    exprs += [
        pl.col(c).cast(pl.Float64).fill_null(999.0).clip(0.0, DAYS_CAP).log1p().alias(c)
        for c in DAYS_COLUMNS
    ]
    exprs += [pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in FLAG_COLUMNS]
    exprs += [pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in PLAIN_COLUMNS]
    exprs += [
        pl.col("pending_to_reallocated_ratio")
        .cast(pl.Float64)
        .fill_null(0.0)
        .clip(0.0)
        .log1p()
        .alias("pending_to_reallocated_ratio"),
        (pl.col("power_on_hours").cast(pl.Float64).fill_null(0.0) / (24 * 365.25)).alias(
            "power_on_years"
        ),
    ]
    return rows.select(exprs)


def feature_columns() -> list[str]:
    """Signature columns in the order `signature_frame` writes them."""
    return [
        *COUNT_COLUMNS,
        *SIGNED_COLUMNS,
        *DAYS_COLUMNS,
        *FLAG_COLUMNS,
        *PLAIN_COLUMNS,
        "pending_to_reallocated_ratio",
        "power_on_years",
    ]


@dataclass
class Scaler:
    """Median / interquartile range per column, clipped to +-10, fitted on the
    validation suspects. A column with no spread falls back to its standard
    deviation, then to 1."""

    center: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        center = np.median(x, axis=0)
        iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        std = x.std(axis=0)
        scale = np.where(iqr > 0, iqr, np.where(std > 0, std, 1.0))
        return cls(center.tolist(), scale.tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - np.asarray(self.center)) / np.asarray(self.scale), -10.0, 10.0)


@dataclass
class ClusterModel:
    """A fitted clustering, enough to assign new drives with numpy only."""

    algorithm: str
    params: dict[str, Any]
    columns: list[str]
    scaler: Scaler
    kind: str  # "centroid" or "gmm"
    centroids: list[list[float]] = field(default_factory=list)
    gmm_weights: list[float] = field(default_factory=list)
    gmm_means: list[list[float]] = field(default_factory=list)
    gmm_covariances: list[list[list[float]]] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    action_kinds: list[str] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.centroids) if self.kind == "centroid" else len(self.gmm_weights)

    def matrix(self, frame: pl.DataFrame) -> np.ndarray:
        return self.scaler.transform(frame.select(self.columns).to_numpy().astype(np.float64))

    def assign_scaled(self, z: np.ndarray) -> np.ndarray:
        if self.kind == "centroid":
            c = np.asarray(self.centroids)
            d = ((z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            return d.argmin(axis=1)
        return _gmm_predict(z, self.gmm_weights, self.gmm_means, self.gmm_covariances)

    def assign(self, frame: pl.DataFrame) -> np.ndarray:
        """Cluster index of each row of a `signature_frame`."""
        return self.assign_scaled(self.matrix(frame))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterModel:
        data = dict(data)
        data["scaler"] = Scaler(**data["scaler"])
        return cls(**data)


def _gmm_predict(z: np.ndarray, weights, means, covariances) -> np.ndarray:
    logp = []
    for w, mu, cov in zip(weights, means, covariances, strict=True):
        cov = np.asarray(cov)
        chol = np.linalg.cholesky(cov)
        diff = np.linalg.solve(chol, (z - np.asarray(mu)).T)
        maha = (diff**2).sum(axis=0)
        logdet = 2 * np.log(np.diag(chol)).sum()
        logp.append(np.log(w) - 0.5 * (maha + logdet))
    return np.vstack(logp).argmax(axis=0)


# --- fitting --------------------------------------------------------------------

ALGORITHMS = ("kmeans", "gmm", "agglomerative", "hdbscan", "spectral")


def fit_labels(algorithm: str, z: np.ndarray, param: int, seed: int = 0) -> tuple[np.ndarray, Any]:
    """Cluster labels of `z` (scaled signature) and the fitted estimator.
    `param` is k, or HDBSCAN's `min_cluster_size`. HDBSCAN's noise (-1)
    becomes its own cluster, the last one."""
    if algorithm == "kmeans":
        est = KMeans(n_clusters=param, n_init=10, random_state=seed).fit(z)
        return est.labels_, est
    if algorithm == "gmm":
        est = GaussianMixture(
            n_components=param, covariance_type="full", reg_covar=1e-3, random_state=seed, n_init=2
        ).fit(z)
        return est.predict(z), est
    if algorithm == "agglomerative":
        est = AgglomerativeClustering(n_clusters=param, linkage="ward").fit(z)
        return est.labels_, est
    if algorithm == "spectral":
        # k-nearest-neighbour affinity graph (15 neighbours). An RBF kernel was
        # tried first and did not converge in 2 minutes on the real suspects: with
        # a local-scale width the graph falls apart into near-disconnected pieces
        # and the eigensolver stalls. On heavy-tailed SMART data this version tends
        # to cut off tiny outlier groups, which the minimum-size rule then excludes.
        est = SpectralClustering(
            n_clusters=param,
            affinity="nearest_neighbors",
            n_neighbors=min(15, len(z) - 1),
            random_state=seed,
            assign_labels="cluster_qr",
        ).fit(z)
        return est.labels_, est
    if algorithm == "hdbscan":
        est = HDBSCAN(min_cluster_size=param).fit(z)
        labels = est.labels_.copy()
        if (labels < 0).any():
            labels[labels < 0] = labels.max() + 1
        return labels, est
    raise ValueError(f"unknown algorithm {algorithm!r}")


def build_model(
    algorithm: str,
    param: int,
    z: np.ndarray,
    labels: np.ndarray,
    est: Any,
    scaler: Scaler,
    columns: list[str],
) -> ClusterModel:
    """A `ClusterModel` from a fit: the mixture's own parameters for GMM,
    otherwise the mean of each cluster's members (nearest-centroid assignment,
    what the agent can run for any algorithm)."""
    params = {"k" if algorithm != "hdbscan" else "min_cluster_size": int(param)}
    if algorithm == "gmm":
        return ClusterModel(
            algorithm,
            params,
            columns,
            scaler,
            "gmm",
            gmm_weights=est.weights_.tolist(),
            gmm_means=est.means_.tolist(),
            gmm_covariances=est.covariances_.tolist(),
        )
    ids = np.unique(labels)
    centroids = [z[labels == i].mean(axis=0).tolist() for i in ids]
    return ClusterModel(algorithm, params, columns, scaler, "centroid", centroids=centroids)


def model_centroids(model: ClusterModel, z: np.ndarray) -> np.ndarray:
    """Mean scaled signature of the drives each cluster is assigned (also for a
    mixture, so every model can be named and plotted the same way)."""
    labels = model.assign_scaled(z)
    return np.vstack(
        [
            z[labels == i].mean(axis=0) if (labels == i).any() else np.zeros(z.shape[1])
            for i in range(model.n_clusters)
        ]
    )


def stability(
    algorithm: str,
    param: int,
    z: np.ndarray,
    scaler: Scaler,
    columns: list[str],
    n_draws: int = 5,
    seed: int = 0,
) -> float:
    """Mean adjusted Rand index between the assignments of the full set by two
    models fitted on disjoint random halves."""
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_draws):
        perm = rng.permutation(len(z))
        halves = perm[: len(z) // 2], perm[len(z) // 2 :]
        assigned = []
        for h in halves:
            labels, est = fit_labels(algorithm, z[h], param, seed=seed + b)
            assigned.append(
                build_model(algorithm, param, z[h], labels, est, scaler, columns).assign_scaled(z)
            )
        scores.append(adjusted_rand_score(*assigned))
    return float(np.mean(scores))


def cross_half_separation(
    clusters: np.ndarray, bands: np.ndarray, failing: np.ndarray, halves: np.ndarray
) -> float:
    """How much clusters separate failure rates *within* each score band,
    measured so that it does not grow with k by chance: cell failure rates are
    estimated on one half of the drives and the deviations from the band rate
    are multiplied across halves (a covariance; noise cancels in expectation),
    averaged over both orders, weighted by drives."""
    total = 0.0
    n = len(clusters)
    for a, b in ((0, 1), (1, 0)):
        for band in np.unique(bands):
            in_band = bands == band
            ma, mb = in_band & (halves == a), in_band & (halves == b)
            if ma.sum() == 0 or mb.sum() == 0:
                continue
            ra, rb = failing[ma].mean(), failing[mb].mean()
            for c in np.unique(clusters[in_band]):
                ca, cb = ma & (clusters == c), mb & (clusters == c)
                if ca.sum() == 0 or cb.sum() == 0:
                    continue
                total += cb.sum() * (failing[ca].mean() - ra) * (failing[cb].mean() - rb)
    return total / (2 * n)


def ablation(
    z: np.ndarray,
    bands: np.ndarray,
    failing: np.ndarray,
    drive_halves: np.ndarray,
    scaler: Scaler,
    columns: list[str],
    *,
    ks: Sequence[int] = range(3, 11),
    hdbscan_sizes: Sequence[int] = (50, 100, 200),
    seed: int = 0,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    done: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Every algorithm and parameter on the validation suspects: separation,
    silhouette, stability, cluster sizes, wall time."""
    rows = []
    grid = [(a, k) for a in ("kmeans", "gmm", "agglomerative", "spectral") for k in ks]
    grid += [("hdbscan", s) for s in hdbscan_sizes]
    finished = {(r["algorithm"], r["param"]): r for r in done}
    for count, (algorithm, param) in enumerate(grid, start=1):
        if (algorithm, int(param)) in finished:
            rows.append(finished[(algorithm, int(param))])
            continue
        t0 = time.perf_counter()
        labels, est = fit_labels(algorithm, z, param, seed=seed)
        model = build_model(algorithm, param, z, labels, est, scaler, columns)
        assigned = model.assign_scaled(z)
        shares = np.bincount(assigned, minlength=model.n_clusters) / len(z)
        noise_share = float((est.labels_ < 0).mean()) if algorithm == "hdbscan" else 0.0
        smallest = float(
            shares[:-1].min() if algorithm == "hdbscan" and noise_share > 0 else shares.min()
        )
        sil = float(silhouette_score(z, assigned)) if len(np.unique(assigned)) > 1 else float("nan")
        rows.append(
            {
                "algorithm": algorithm,
                "param": int(param),
                "clusters": int(model.n_clusters),
                "separation": cross_half_separation(assigned, bands, failing, drive_halves),
                "silhouette": sil,
                "stability_ari": stability(algorithm, param, z, scaler, columns, seed=seed),
                "smallest_cluster_share": smallest,
                "noise_share": noise_share,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        if progress is not None:
            progress(count, len(grid), rows[-1])
    return rows


def select(
    rows: list[dict[str, Any]], *, min_stability: float = 0.7, min_share: float = 0.02
) -> dict[str, Any]:
    """The pre-registered choice: the largest separation among candidates with
    stability >= `min_stability` and every (non-noise) cluster >= `min_share`
    of suspects; ties go to k-means. If none qualifies, the most stable
    candidate meeting the size floor, and the result says so."""
    ok = [
        r
        for r in rows
        if r["stability_ari"] >= min_stability and r["smallest_cluster_share"] >= min_share
    ]
    if ok:
        best = max(ok, key=lambda r: (round(r["separation"], 9), r["algorithm"] == "kmeans"))
        return {**best, "rule_met": True}
    sized = [r for r in rows if r["smallest_cluster_share"] >= min_share] or rows
    best = max(sized, key=lambda r: r["stability_ari"])
    return {**best, "rule_met": False}


# --- naming -----------------------------------------------------------------------


def _readable(col: str) -> str:
    for suffix, label in (
        ("_30d_delta", " rising"),
        ("_30d_slope", " rising"),
        ("_zero_to_nonzero", " newly non-zero"),
        ("_days_since_last_increase", " quiet for long"),
    ):
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            return READABLE.get(base, base.replace("_", " ")) + label
    return READABLE.get(col, col.replace("_", " "))


def name_clusters(
    centroids: np.ndarray, columns: list[str], top: int = 2
) -> tuple[list[str], list[str]]:
    """A name per cluster from its largest standardised features, and an action
    kind: "inspect connection" when the top features are interface errors,
    "migrate data" when they are pending / uncorrectable sectors, "replace"
    for other damage, "watch" when nothing stands out."""
    names, kinds = [], []
    for c in centroids:
        order = np.argsort(-c)
        picked: list[str] = []
        labels_seen: set[str] = set()
        for i in order:
            if c[i] <= 0.5 or len(picked) == top:
                break
            label = _readable(columns[i])
            if label in labels_seen:  # a counter's change and its slope read the same
                continue
            labels_seen.add(label)
            picked.append(columns[i])
        if not picked:
            names.append("no clear damage")
            kinds.append("watch")
            continue
        names.append(", ".join(_readable(p) for p in picked))
        bases = {p.split("_30d")[0].split("_days_since")[0].split("_zero_to")[0] for p in picked}
        if bases & set(INTERFACE_FEATURES):
            kinds.append("inspect connection")
        elif bases & set(MEDIA_AT_RISK_FEATURES):
            kinds.append("migrate data")
        elif bases <= {"power_on_years", "temperature_celsius", "telemetry_coverage_30d"}:
            kinds.append("watch")
        else:
            kinds.append("replace")
    # Make names unique so the matrix rows can be told apart.
    seen: dict[str, int] = {}
    unique = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        unique.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return unique, kinds
