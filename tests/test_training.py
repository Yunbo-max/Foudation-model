"""Trainer integration fixtures test reproducibility, not training quality."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class CommandTests(unittest.TestCase):
    def test_training_and_generation_help(self):
        # Help must work even when optional PyTorch is not installed.
        for script in ("train.py", "generate.py"):
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: training/resume were not tested")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        import numpy as np
        from fm_tutorial.tokenizer import ByteBPETokenizer
        self.torch = torch
        self.np = np
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.path = Path(self.tmp.name)
        self.data = self.path / "data"
        self.data.mkdir()
        # Arbitrary token ids are an intentionally small program-regression fixture.
        tokenizer = ByteBPETokenizer.train(["正确性测试。"], vocab_size=258)
        tokenizer.save(self.data / "tokenizer.json")
        np.array(list(range(30, 94)) * 2, dtype=np.uint32).tofile(self.data / "train.bin")
        np.array(list(range(40, 88)), dtype=np.uint32).tofile(self.data / "val.bin")
        self.config = self.path / "tiny.json"
        self.config.write_text(json.dumps({
            "model": {"vocab_size": 8192, "context_length": 8, "d_model": 16,
                "n_layers": 1, "n_heads": 4, "n_kv_heads": 2, "d_ff": 32, "dropout": 0.1},
            "training": {"batch_size": 2, "gradient_accumulation": 2, "learning_rate": 0.001,
                "weight_decay": 0.01, "warmup_steps": 1, "max_steps": 5,
                "min_lr_ratio": 0.1, "eval_every": 2, "eval_batches": 2,
                "gradient_clip": 1.0, "seed": 123, "mixed_precision": False}
        }), encoding="utf-8")

    def trainer(self):
        spec = importlib.util.spec_from_file_location("tutorial_train", ROOT / "scripts" / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_resume_matches_uninterrupted_optimizer_updates(self):
        trainer = self.trainer()
        trainer.train(self.config, self.data, self.path / "full", steps=3, device="cpu")
        trainer.train(self.config, self.data, self.path / "split", steps=1, device="cpu")
        trainer.train(self.config, self.data, self.path / "split", steps=3, device="cpu",
                      resume=self.path / "split" / "last.pt")
        full = self.torch.load(self.path / "full" / "last.pt", weights_only=False)
        split = self.torch.load(self.path / "split" / "last.pt", weights_only=False)
        self.assertEqual(split["step"], 3)
        self.assertEqual(split["tokens_seen"], 84)
        self.assertEqual(split["model_config"]["vocab_size"], 258)
        for name in full["model_state"]:
            self.torch.testing.assert_close(full["model_state"][name], split["model_state"][name],
                                           rtol=0, atol=0)
        for param_id, state in full["optimizer_state"]["state"].items():
            for key, value in state.items():
                self.torch.testing.assert_close(value, split["optimizer_state"]["state"][param_id][key],
                                               rtol=0, atol=0)
        self.assertEqual(full["rng"]["python"], split["rng"]["python"])
        self.assertEqual(full["data_fingerprint"], split["data_fingerprint"])

    def test_resume_rejects_changed_data(self):
        trainer = self.trainer()
        trainer.train(self.config, self.data, self.path / "run", steps=1, device="cpu")
        with (self.data / "train.bin").open("ab") as handle:
            handle.write(self.np.array([99], dtype=self.np.uint32).tobytes())
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            trainer.train(self.config, self.data, self.path / "run", steps=2, device="cpu",
                          resume=self.path / "run" / "last.pt")

    def test_resume_rejects_changed_hyperparameters(self):
        trainer = self.trainer()
        trainer.train(self.config, self.data, self.path / "run", steps=1, device="cpu")
        raw = json.loads(self.config.read_text(encoding="utf-8"))
        raw["training"]["learning_rate"] = 0.002
        self.config.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration"):
            trainer.train(self.config, self.data, self.path / "run", steps=2, device="cpu",
                          resume=self.path / "run" / "last.pt")

    def test_completed_resume_still_exports_checkpoint_to_new_directory(self):
        trainer = self.trainer()
        trainer.train(self.config, self.data, self.path / "source", steps=1, device="cpu")
        result = trainer.train(self.config, self.data, self.path / "copy", steps=1, device="cpu",
                               resume=self.path / "source" / "last.pt")
        self.assertTrue(Path(result["checkpoint"]).is_file())
        self.assertEqual(self.torch.load(result["checkpoint"], weights_only=True)["step"], 1)

    def test_generation_cli_loads_checkpoint_and_checks_tokenizer_identity(self):
        self.trainer().train(self.config, self.data, self.path / "run", steps=1, device="cpu")
        command = [sys.executable, str(ROOT / "scripts" / "generate.py"),
                   "--checkpoint", str(self.path / "run" / "last.pt"),
                   "--tokenizer", str(self.data / "tokenizer.json"), "--prompt", "abc",
                   "--max-new-tokens", "2", "--temperature", "0"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("abc"))
        with (self.data / "tokenizer.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fingerprint", result.stderr)

    def test_too_short_or_padding_corpus_is_rejected(self):
        trainer = self.trainer()
        for tokens in ([31, 32], [31, 32, 257, 34, 35, 36, 37, 38]):
            self.np.array(tokens, dtype=self.np.uint32).tofile(self.data / "train.bin")
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                trainer.train(self.config, self.data, self.path / "run", steps=1, device="cpu")

    def test_generation_masks_pad_and_stops_at_eos(self):
        from fm_tutorial.model import ModelConfig, TransformerLM
        spec = importlib.util.spec_from_file_location("tutorial_generate", ROOT / "scripts" / "generate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = TransformerLM(ModelConfig(vocab_size=258, context_length=2, d_model=8,
            n_layers=1, n_heads=2, n_kv_heads=1, d_ff=16, tie_embeddings=False))
        with self.torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.token_embedding.weight[42].fill_(1.0)
            model.final_norm.weight.fill_(1.0)
            model.lm_head.weight[257].fill_(10.0)
            model.lm_head.weight[256].fill_(1.0)
        # The prompt exceeds context; only the model input window is truncated.
        ids = module.generate_tokens(model, [42, 42, 42], max_new_tokens=5, temperature=0.0)
        self.assertEqual(ids, [42, 42, 42, 256])


if __name__ == "__main__":
    unittest.main()
