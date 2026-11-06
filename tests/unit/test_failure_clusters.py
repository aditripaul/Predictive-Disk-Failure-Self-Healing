import numpy as np
import polars as pl
import pytest

from src.models.failure_clusters import (
    DAYS_CAP,
    SOURCE_COLUMNS,
    ClusterModel,
    Scaler,
    ablation,
    build_model,
    cross_half_separation,
    feature_columns,
    fit_labels,
    name_clusters,
    select,
    signature_frame,
)


def _gold(n: int = 10, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.random(n).astype(np.float32) for c in SOURCE_COLUMNS}
    for c in SOURCE_COLUMNS:
        if c.endswith("_zero_to_nonzero"):
            data[c] = rng.random(n) < 0.5
        if c.endswith("_days_since_last_increase"):
            data[c] = np.full(n, 999, dtype=np.int16)
    df = pl.DataFrame(data).with_columns(
        drive_id=pl.Series([f"d{i}" for i in range(n)]),
        date=pl.lit(None, dtype=pl.Date),
    )
    return df


def test_signature_frame_transforms_and_fills():
    gold = _gold().with_columns(
        pl.lit(None, dtype=pl.Float32).alias("current_pending_sector_count"),
        pl.lit(-8.0, dtype=pl.Float32).alias("reallocated_sector_count_30d_delta"),
        pl.lit(8766.0, dtype=pl.Float32).alias("power_on_hours"),
    )
    sig = signature_frame(gold)
    assert sig.columns == ["drive_id", "date", *feature_columns()]
    assert (sig["current_pending_sector_count"] == 0).all()  # missing = nothing happened
    np.testing.assert_allclose(sig["reallocated_sector_count_30d_delta"], -np.log1p(8.0), rtol=1e-6)
    np.testing.assert_allclose(
        sig["reallocated_sector_count_days_since_last_increase"], np.log1p(DAYS_CAP)
    )
    np.testing.assert_allclose(sig["power_on_years"], 8766 / (24 * 365.25), rtol=1e-6)
    assert sig.select(feature_columns()).null_count().sum_horizontal().item() == 0


def _blobs(n_per: int = 60, k: int = 4, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8.0, size=(k, d))
    x = np.vstack([c + rng.normal(size=(n_per, d)) for c in centers])
    truth = np.repeat(np.arange(k), n_per)
    return x, truth


@pytest.mark.parametrize("algorithm", ["kmeans", "gmm", "agglomerative", "spectral"])
def test_models_recover_clear_groups_and_round_trip(algorithm):
    x, truth = _blobs()
    scaler = Scaler.fit(x)
    z = scaler.transform(x)
    cols = [f"f{i}" for i in range(x.shape[1])]
    labels, est = fit_labels(algorithm, z, 4)
    model = build_model(algorithm, 4, z, labels, est, scaler, cols)
    assigned = model.assign_scaled(z)
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(truth, assigned) > 0.95
    again = ClusterModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(again.assign_scaled(z), assigned)


def test_kmeans_assignment_equals_sklearn_predict():
    x, _ = _blobs()
    scaler = Scaler.fit(x)
    z = scaler.transform(x)
    labels, est = fit_labels("kmeans", z, 4)
    model = build_model("kmeans", 4, z, labels, est, scaler, [f"f{i}" for i in range(z.shape[1])])
    np.testing.assert_array_equal(
        model.assign_scaled(z), np.unique(est.predict(z), return_inverse=True)[1]
    )


def test_hdbscan_noise_becomes_its_own_cluster():
    x, _ = _blobs()
    x = np.vstack([x, np.random.default_rng(1).uniform(-40, 40, size=(15, x.shape[1]))])
    z = Scaler.fit(x).transform(x)
    labels, est = fit_labels("hdbscan", z, 20)
    assert labels.min() >= 0
    if (est.labels_ < 0).any():
        assert labels[est.labels_ < 0].max() == labels.max()


def test_separation_is_zero_for_random_clusters_and_positive_for_informative_ones():
    rng = np.random.default_rng(0)
    n = 4000
    bands = rng.choice(["warn", "cordon"], size=n)
    clusters = rng.integers(0, 4, size=n)
    halves = rng.integers(0, 2, size=n)
    # Cluster 3 fails far more often within each band.
    failing = rng.random(n) < np.where(clusters == 3, 0.6, 0.1)
    informative = cross_half_separation(clusters, bands, failing, halves)
    shuffled = cross_half_separation(rng.permutation(clusters), bands, failing, halves)
    assert informative > 0.01
    assert abs(shuffled) < informative / 5


def test_ablation_and_selection_rule():
    x, truth = _blobs(n_per=40)
    scaler = Scaler.fit(x)
    z = scaler.transform(x)
    rng = np.random.default_rng(0)
    failing = rng.random(len(z)) < np.where(truth == 0, 0.7, 0.1)
    bands = np.array(["warn"] * len(z))
    halves = rng.integers(0, 2, size=len(z))
    rows = ablation(
        z,
        bands,
        failing,
        halves,
        scaler,
        [f"f{i}" for i in range(z.shape[1])],
        ks=(3, 4),
        hdbscan_sizes=(10,),
    )
    assert {r["algorithm"] for r in rows} == {
        "kmeans",
        "gmm",
        "agglomerative",
        "spectral",
        "hdbscan",
    }
    chosen = select(rows)
    assert chosen["rule_met"] and chosen["stability_ari"] >= 0.7
    # Nothing stable enough: the most stable candidate is taken and flagged.
    weak = [{**r, "stability_ari": 0.1 + 0.01 * i} for i, r in enumerate(rows)]
    assert select(weak)["rule_met"] is False


def test_cluster_names_and_action_kinds():
    cols = [
        "udma_crc_error_count",
        "current_pending_sector_count",
        "reallocated_sector_count",
        "power_on_years",
    ]
    centroids = np.array(
        [
            [3.0, 0.1, 0.0, 0.0],  # interface errors
            [0.0, 4.0, 1.0, 0.0],  # pending sectors
            [0.0, 0.0, 3.0, 0.0],  # reallocations
            [0.0, 0.0, 0.0, 0.2],  # nothing stands out
        ]
    )
    names, kinds = name_clusters(centroids, cols)
    assert kinds == ["inspect connection", "migrate data", "replace", "watch"]
    assert names[0].startswith("CRC errors") and names[3] == "no clear damage"
    assert len(set(names)) == len(names)
