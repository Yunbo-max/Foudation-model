"""Tiny correctness fixtures; these are not model-quality benchmarks."""
import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: model numerics were not tested")
class TransformerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from fm_tutorial.model import ModelConfig, TransformerLM
        self.torch = torch
        self.Config = ModelConfig
        torch.manual_seed(17)
        self.model = TransformerLM(ModelConfig(vocab_size=300, context_length=8,
            d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64)).eval()

    def test_shape_and_tied_parameter_count(self):
        # An untied head or wrong GQA projection size would change this count.
        result = self.model(self.torch.tensor([[11, 12, 13, 14]]))
        self.assertEqual(tuple(result["logits"].shape), (1, 4, 300))
        self.assertIsNone(result["loss"])
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 28192)

    def test_future_tokens_cannot_change_prefix(self):
        torch = self.torch
        a = torch.tensor([[11, 12, 13, 14, 15, 16]])
        b = torch.tensor([[11, 12, 13, 90, 91, 92]])
        with torch.no_grad():
            left, right = self.model(a)["logits"], self.model(b)["logits"]
        torch.testing.assert_close(left[:, :3], right[:, :3], atol=1e-6, rtol=1e-6)

    def test_loss_shifts_targets_and_ignores_pad(self):
        torch = self.torch
        x = torch.tensor([[11, 12, 13, 14]])
        labels = torch.tensor([[11, 12, 257, 14]])
        result = self.model(x, labels=labels)
        logits = result["logits"]
        expected = (-torch.log_softmax(logits[0, 0], -1)[12]
                    - torch.log_softmax(logits[0, 2], -1)[14]) / 2
        torch.testing.assert_close(result["loss"], expected)

    def test_backward_reaches_attention_and_embedding(self):
        self.model.train()
        x = self.torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
        self.model(x, labels=x)["loss"].backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(self.torch.isfinite(parameter.grad).all().item(), name)
        self.assertGreater(self.model.blocks[0].attention.q_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.token_embedding.weight.grad.abs().sum().item(), 0)

    def test_all_padding_targets_return_differentiable_zero(self):
        x = self.torch.tensor([[11, 12, 13]])
        loss = self.model(x, labels=self.torch.full_like(x, 257))["loss"]
        self.assertEqual(loss.item(), 0.0)
        loss.backward()

    def test_all_padding_zero_loss_cannot_overflow_under_autocast(self):
        from fm_tutorial.model import TransformerLM
        torch = self.torch
        model = TransformerLM(self.Config(vocab_size=258, context_length=2, d_model=8,
            n_layers=1, n_heads=2, n_kv_heads=1, d_ff=16, tie_embeddings=False))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.token_embedding.weight[11:13].fill_(1.0)
            model.final_norm.weight.fill_(1.0)
            model.lm_head.weight.fill_(100.0)
        x = torch.tensor([[11, 12]])
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            result = model(x, labels=torch.full_like(x, 257))
        self.assertTrue(torch.isfinite(result["logits"]).all().item())
        # Finite individual logits can still overflow a float16 sum before *0.
        self.assertEqual(result["loss"].item(), 0.0)
        result["loss"].backward()
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_invalid_dimensions_and_context_are_rejected(self):
        for kwargs in ({"d_model": 30}, {"n_kv_heads": 3},
                       {"d_model": 12, "n_heads": 4}, {"vocab_size": 257}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.Config(vocab_size=kwargs.pop("vocab_size", 300), **kwargs)
        with self.assertRaises(ValueError):
            self.model(self.torch.ones((1, 9), dtype=self.torch.long))
        with self.assertRaises(ValueError):
            self.model(self.torch.ones((1, 3), dtype=self.torch.long),
                       labels=self.torch.ones((1, 2), dtype=self.torch.long))

    def test_rmsnorm_reduces_in_float32(self):
        from fm_tutorial.model import RMSNorm
        # Squaring large fp16 values without promotion would overflow.
        x = self.torch.tensor([[1000.0, -1000.0]], dtype=self.torch.float16)
        y = RMSNorm(2)(x)
        self.torch.testing.assert_close(y.float(), self.torch.tensor([[1.0, -1.0]]),
                                       rtol=1e-3, atol=1e-3)

    def test_attention_scores_stay_finite_under_float16_autocast(self):
        from fm_tutorial.model import CausalAttention
        torch = self.torch
        attention = CausalAttention(self.Config(vocab_size=258, context_length=2,
            d_model=8, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=16)).eval()
        with torch.no_grad():
            attention.q_proj.weight.fill_(1.0)
            attention.k_proj.weight.fill_(1.0)
            attention.v_proj.weight.fill_(0.01)
            attention.out_proj.weight.fill_(0.01)
        x = torch.full((1, 2, 8), 100.0)
        valid_keys = torch.ones((1, 2), dtype=torch.bool)
        with torch.no_grad():
            reference = attention(x, valid_keys)
            with torch.autocast(device_type="cpu", dtype=torch.float16):
                output = attention(x, valid_keys)
        # Projections fit in fp16, but QK^T scores exceed its finite range.
        # Float32 score matmul/softmax must survive ambient autocast.
        self.assertTrue(torch.isfinite(reference).all().item())
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(output.dtype, torch.float16)
        torch.testing.assert_close(output.float(), reference, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
