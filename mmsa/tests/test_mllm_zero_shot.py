import json
import tempfile
import unittest
from pathlib import Path

from mmsa.data.mllm_zero_shot_baseline import _load_checkpoint, _parse_score, _stable_hash


class ZeroShotHelpersTest(unittest.TestCase):
    def test_parse_score(self):
        cases = [
            ("1.4", 1.4, True),
            ("Score: -0.8", -0.8, True),
            ("4.2", 3.0, True),
            ("no numeric answer", 0.0, False),
        ]
        for reply, expected, ok in cases:
            with self.subTest(reply=reply):
                score, parsed = _parse_score(reply, -3.0, 3.0)
                self.assertAlmostEqual(score, expected)
                self.assertIs(parsed, ok)

    def test_checkpoint_rejects_different_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            good_hash = _stable_hash({"model": "a"})
            path.write_text(
                json.dumps({"config_hash": good_hash, "i": 0, "score": 0.0, "parsed_ok": True}) + "\n",
                encoding="utf-8",
            )
            self.assertIn(0, _load_checkpoint(path, good_hash, 1))
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                _load_checkpoint(path, _stable_hash({"model": "b"}), 1)


if __name__ == "__main__":
    unittest.main()