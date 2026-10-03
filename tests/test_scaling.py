"""Analytic arithmetic diagnostics; these are not measured scaling results."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from fm_tutorial.scaling import fit_power_law, read_measurements


class ScalingTests(unittest.TestCase):
    def test_recovers_known_log_linear_identity_and_evaluates_holdout(self):
        # Hand-computable identity loss = 8 / sqrt(tokens).
        rows = [
            {"params": 10, "tokens": 1, "loss": 8, "split": "train"},
            {"params": 10, "tokens": 4, "loss": 4, "split": "train"},
            {"params": 10, "tokens": 16, "loss": 2, "split": "train"},
            {"params": 10, "tokens": 64, "loss": 1, "split": "holdout"},
        ]
        result = fit_power_law(rows, axis="tokens")
        self.assertAlmostEqual(result["exponent"], -0.5, places=12)
        self.assertAlmostEqual(result["coefficient"], 8, places=12)
        self.assertAlmostEqual(result["holdout"][0]["prediction"], 1, places=12)
        self.assertLess(result["holdout_log_rmse"], 1e-12)
        # A different held-out answer must not change fitted parameters.
        rows[-1]["loss"] = 9
        alternate = fit_power_law(rows, axis="tokens")
        self.assertEqual(result["exponent"], alternate["exponent"])
        self.assertGreater(alternate["holdout_log_rmse"], 1)

    def test_rejects_unidentifiable_or_confounded_design(self):
        cases = [
            [{"params": 1, "tokens": 2, "loss": 1}] * 4,
            [{"params": i, "tokens": i, "loss": 1 / i} for i in range(1, 6)],
            [{"params": 1, "tokens": i, "loss": 0} for i in range(1, 6)],
            [{"params": 1, "tokens": i, "loss": math.nan} for i in range(1, 6)],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    fit_power_law(rows)

    def test_csv_requires_measurement_fields_and_accepts_explicit_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.csv"
            path.write_text("params,tokens,loss,split\n10,1,8,train\n10,4,4,train\n10,16,2,train\n10,64,1,holdout\n")
            rows = read_measurements(path)
            self.assertEqual(len(rows), 4)
            self.assertEqual(fit_power_law(rows)["train_count"], 3)
            path.write_text("tokens,loss\n1,1\n")
            with self.assertRaisesRegex(ValueError, "params"):
                read_measurements(path)


if __name__ == "__main__":
    unittest.main()
