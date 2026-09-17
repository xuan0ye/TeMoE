from __future__ import annotations

import json
import pickle

import numpy as np

from mmsa.analysis.official_crossing import prepare_crossing


def test_prepare_official_crossing_is_dimension_matched(tmp_path):
    rng = np.random.default_rng(7)
    payload = {}
    counts = {"train": 4, "valid": 2, "test": 3}
    for split, count in counts.items():
        payload[split] = {
            "text": rng.normal(size=(count, 5, 6)).astype(np.float32),
            "audio": rng.normal(size=(count, 5, 2)).astype(np.float32),
            "vision": rng.normal(size=(count, 5, 3)).astype(np.float32),
            "regression_labels": rng.normal(size=(count, 1)).astype(np.float32),
        }

    source = tmp_path / "toy.pkl"
    with source.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = {}
    for split, count in counts.items():
        expected[split] = rng.normal(size=(count, 4)).astype(np.float32)
        np.save(cache / f"{split}.npy", expected[split])

    output = tmp_path / "prepared"
    manifest = prepare_crossing(str(source), str(cache), str(output))
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved["source_pkl_sha256"] == manifest["source_pkl_sha256"]

    arms = {}
    for arm in ("cached", "no_cache"):
        with open(saved["arms"][arm]["feature_path"], "rb") as handle:
            arms[arm] = pickle.load(handle)
        assert arms[arm]["train"]["text"].shape == (4, 5, 10)
        assert arms[arm]["train"]["text_lengths"].tolist() == [5, 5, 5, 5]

    np.testing.assert_array_equal(
        arms["cached"]["train"]["text"][..., :6],
        arms["no_cache"]["train"]["text"][..., :6],
    )
    np.testing.assert_array_equal(arms["no_cache"]["train"]["text"][..., 6:], 0.0)
    for index in range(5):
        np.testing.assert_allclose(
            arms["cached"]["train"]["text"][:, index, 6:],
            expected["train"],
        )
