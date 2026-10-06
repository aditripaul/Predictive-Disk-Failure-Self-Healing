import numpy as np
import polars as pl

from pipelines.experiment_model import _predict_excluding
from pipelines.train_model import _build_feature_arrays
from src.models.training import predict_proba_positive, train_lightgbm


def _parts(tmp_path, sizes=(5, 1, 7)):
    rng = np.random.default_rng(0)
    paths = []
    for i, n in enumerate(sizes):
        path = tmp_path / f"part_{i}.parquet"
        pl.DataFrame(
            {
                "f1": rng.normal(size=n),
                "f2": [None if j % 3 == 0 else float(j) for j in range(n)],
                "label": rng.integers(0, 2, size=n),
            }
        ).write_parquet(path)
        paths.append(path)
    return paths


def test_memory_mapped_matrix_equals_the_in_ram_matrix(tmp_path):
    parts = _parts(tmp_path)
    x_ram, y_ram = _build_feature_arrays(parts, ["f1", "f2"])
    out = tmp_path / "x.npy"
    x_map, y_map = _build_feature_arrays(parts, ["f1", "f2"], out_path=out)
    assert isinstance(x_map, np.memmap)
    np.testing.assert_array_equal(x_ram, np.asarray(x_map))
    np.testing.assert_array_equal(y_ram, y_map)
    del x_map
    reloaded = np.load(out, mmap_mode="r")  # a valid .npy file, readable read-only
    np.testing.assert_array_equal(x_ram, np.asarray(reloaded))
    assert reloaded.dtype == np.float32


def test_memory_mapped_matrix_with_no_parts_is_an_empty_valid_file(tmp_path):
    x, y = _build_feature_arrays([], ["f1", "f2"], out_path=tmp_path / "x.npy")
    assert x.shape == (0, 2) and len(y) == 0
    assert np.load(tmp_path / "x.npy").shape == (0, 2)


def _model_and_data(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 4)).astype(np.float32)
    y = (x[:, 0] + rng.normal(scale=0.5, size=600) > 0.8).astype(int)
    model = train_lightgbm(x, y, params={"n_estimators": 20})
    path = tmp_path / "x_val.npy"
    np.save(path, x)
    return model, x, np.load(path, mmap_mode="r")


def test_chunked_scoring_of_a_read_only_map_equals_one_shot_scoring(tmp_path):
    model, x, mapped = _model_and_data(tmp_path)
    expected = np.asarray(model.predict_proba(x))[:, 1]
    np.testing.assert_allclose(predict_proba_positive(model, mapped, chunk_rows=97), expected)
    np.testing.assert_allclose(predict_proba_positive(model, mapped), expected)
    assert predict_proba_positive(model, x[:0]).shape == (0,)


def test_predict_excluding_zeroes_columns_without_touching_the_map(tmp_path, monkeypatch):
    import pipelines.experiment_model as experiment_model

    model, x, mapped = _model_and_data(tmp_path)
    zeroed = x.copy()
    zeroed[:, [0, 2]] = 0.0
    expected = np.asarray(model.predict_proba(zeroed))[:, 1]
    monkeypatch.setattr(experiment_model, "PREDICT_CHUNK_ROWS", 113)
    np.testing.assert_allclose(_predict_excluding(model, mapped, [0, 2]), expected)
    np.testing.assert_array_equal(np.asarray(mapped), x)  # the file is unchanged
    np.testing.assert_allclose(
        _predict_excluding(model, mapped, []), np.asarray(model.predict_proba(x))[:, 1]
    )
