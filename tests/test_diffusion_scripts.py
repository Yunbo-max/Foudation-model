"""CLI integration checks; tiny synthetic inputs are not quality benchmarks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiffusionScriptTests(unittest.TestCase):
    def test_public_imports_support_the_chapter_examples(self):
        from fm_tutorial.diffusion import GaussianDiffusion, TimeMLP, VPSDE
        import torch
        process = GaussianDiffusion(steps=10)
        model = TimeMLP(2)
        x0, t = torch.zeros(2, 2), torch.tensor([2, 7])
        xt = process.q_sample(x0, t)
        self.assertEqual(model(xt, (t.float() + 1) / process.steps).shape, x0.shape)
        self.assertEqual(VPSDE().marginal(x0, torch.tensor([0.2, 0.7])).shape, x0.shape)

    def run_script(self, script, *args):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                                cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_gaussian_demo_writes_finite_generated_samples_and_run_record(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            self.run_script("diffusion_demo.py", "--mode", "gaussian", "--sampler", "ddim",
                            "--train-steps", "3", "--diffusion-steps", "10",
                            "--sample-steps", "4", "--samples", "8", "--out-dir", directory)
            samples = np.load(Path(directory) / "samples.npy")
            self.assertEqual(samples.shape, (8, 2))
            self.assertTrue(np.isfinite(samples).all())
            record = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(record["mode"], "gaussian")
            self.assertIn("synthetic", record["data_source"])
            self.assertEqual(record["train_steps"], 3)
            self.assertTrue((Path(directory) / "training.csv").is_file())

    def test_supplied_array_retains_nonbatch_image_shape(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.npy"
            np.save(source, np.zeros((4, 1, 2, 2), dtype=np.float32))
            output = str(Path(directory) / "out")
            self.run_script("diffusion_demo.py", "--mode", "gaussian", "--data-npy", str(source),
                            "--train-steps", "1", "--diffusion-steps", "5",
                            "--sample-steps", "2", "--samples", "2", "--out-dir", output)
            self.assertEqual(np.load(Path(output) / "samples.npy").shape, (2, 1, 2, 2))

    def test_score_ode_runs_reverse_time_without_nan(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            self.run_script("diffusion_demo.py", "--mode", "score", "--sampler", "ode",
                            "--train-steps", "2", "--sample-steps", "8",
                            "--samples", "4", "--out-dir", directory)
            self.assertTrue(np.isfinite(np.load(Path(directory) / "samples.npy")).all())

    def test_categorical_text_sampler_emits_only_ordinary_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_script("diffusion_demo.py", "--mode", "categorical", "--train-steps", "2",
                            "--diffusion-steps", "5", "--samples", "2", "--out-dir", directory)
            record = json.loads((Path(directory) / "report.json").read_text())
            self.assertNotIn("<mask>", record["vocabulary"])

    def test_masked_demo_generates_without_mask_and_preserves_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_script("diffusion_demo.py", "--mode", "masked", "--train-steps", "2",
                            "--sample-steps", "4", "--samples", "2", "--out-dir", directory)
            lines = (Path(directory) / "samples.txt").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(line.startswith("the small ") for line in lines))
            self.assertTrue(all("<mask>" not in line for line in lines))

    def test_modality_shapes_use_same_forward_law_and_report_oracle_check(self):
        result = self.run_script("diffusion_shapes.py")
        report = json.loads(result.stdout)
        self.assertEqual(set(report["continuous"]),
                         {"text_embeddings", "image_latent", "table_numeric", "video_latent", "4d_trajectories"})
        self.assertLess(max(item["oracle_max_error"] for item in report["continuous"].values()), 1e-4)
        self.assertTrue(report["table_categorical"]["posterior_normalized"])
        self.assertTrue(report["masked_text"]["condition_preserved"])

    def test_conditioning_check_verifies_zero_initialization_and_cfg_anchors(self):
        result = self.run_script("conditioning_check.py")
        report = json.loads(result.stdout)
        self.assertEqual(report["adaln_zero_identity_error"], 0.0)
        self.assertEqual(report["zero_spatial_identity_error"], 0.0)
        self.assertTrue(report["cfg_scale_zero_matches_unconditional"])
        self.assertTrue(report["cfg_scale_one_matches_conditional"])
        self.assertLess(report["concat_projected_sum_error"], 1e-5)
        self.assertGreater(report["cross_attention_condition_change"], 0)

    def test_conditional_demo_trains_null_branch_and_saves_paired_outputs(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            self.run_script("conditioning_demo.py", "--method", "adaln", "--train-steps", "3",
                            "--samples", "4", "--diffusion-steps", "10", "--sample-steps", "4",
                            "--out-dir", directory)
            values = np.load(Path(directory) / "samples.npy")
            self.assertEqual(values.shape, (2, 4, 2))
            self.assertTrue(np.isfinite(values).all())
            record = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(record["method"], "adaln")
            self.assertGreater(record["null_condition_training_examples"], 0)
            self.assertGreater(record["conditional_training_examples"], 0)
            self.assertEqual(record["schedule"]["name"], "linear")
            self.assertEqual(record["schedule"]["beta_end"], 0.12)
            self.assertTrue(record["paired_initial_noise_equal"])

    def test_conditional_demo_requires_actual_training_for_both_cfg_branches(self):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory() as directory:
            for dropout in ("1e-12", "0.999999999999"):
                with self.subTest(dropout=dropout):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/conditioning_demo.py"),
                         "--train-steps", "1", "--condition-dropout", dropout,
                         "--out-dir", directory], cwd=ROOT, env=env,
                        text=True, capture_output=True, timeout=60)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Both null and conditional training examples are required", result.stderr)
                    self.assertFalse((Path(directory) / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
