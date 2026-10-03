"""Small hand-calculated loss checks; these are not training benchmarks."""
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for alignment numerical tests")
class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("fm_tutorial.alignment"), "alignment implementation is missing")
        self.alignment = importlib.import_module("fm_tutorial.alignment")

    def completion_fixture(self):
        # Both sequence probabilities are hand-selected. Last logits are unused.
        probabilities = torch.tensor([
            [[.8, .2], [.25, .75], [.6, .4], [.9, .1]],
            [[.3, .7], [.4, .6], [.2, .8], [.5, .5]],
        ], dtype=torch.float64)
        logits = probabilities.log().requires_grad_()
        ids = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])
        mask = torch.tensor([[False, False, True, True], [False, True, False, False]])
        return logits, ids, mask

    def test_completion_log_probs_shift_targets_and_sum(self):
        logits, ids, mask = self.completion_fixture()
        result = self.alignment.completion_log_probs(logits, ids, mask)
        expected = torch.tensor([math.log(.75) + math.log(.6), math.log(.3)], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_long_half_precision_completions_accumulate_in_fp32_and_feed_dpo(self):
        # Uniform-token log p is -log(4096); 8192 targets exceed fp16 range.
        target_count, vocab_size = 8192, 4096
        ids = torch.zeros(1, target_count + 1, dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[:, 0] = False
        expected = -target_count * math.log(vocab_size)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.zeros(1, target_count + 1, vocab_size, dtype=dtype)
                summed = self.alignment.completion_log_probs(logits, ids, mask)
                self.assertTrue(torch.isfinite(summed).all().item(), "long sequence sum overflowed")
                self.assertEqual(summed.dtype, torch.float32)
                self.assertAlmostEqual(summed.item(), expected, delta=.02)
                loss = self.alignment.dpo_loss(summed, summed, summed, summed)
                self.assertAlmostEqual(loss.item(), math.log(2), places=6)

    def test_half_precision_sft_uses_fp32_math_and_keeps_input_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.zeros(1, 3, 2, dtype=dtype, requires_grad=True)
                ids = torch.tensor([[0, 1, 0]])
                mask = torch.tensor([[False, False, True]])
                loss = self.alignment.sft_loss(logits, ids, mask)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertAlmostEqual(loss.item(), math.log(2), places=6)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all().item())
                self.assertEqual(logits.grad.dtype, dtype)
                torch.testing.assert_close(logits.grad[0, 1], torch.tensor([-.5, .5], dtype=dtype))
                self.assertEqual(logits.grad[0, (0, 2)].abs().sum().item(), 0)

    def test_sft_averages_active_tokens_and_masks_gradients(self):
        logits, ids, mask = self.completion_fixture()
        loss = self.alignment.sft_loss(logits, ids, mask)
        self.assertAlmostEqual(loss.item(), -(math.log(.75) + math.log(.6) + math.log(.3)) / 3)
        loss.backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1, 1:].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1:3].abs().sum().item(), 0)

    def test_empty_completion_has_zero_sequence_sum_but_sft_rejects(self):
        logits, ids, mask = self.completion_fixture()
        mask.fill_(False)
        torch.testing.assert_close(self.alignment.completion_log_probs(logits, ids, mask), torch.zeros(2, dtype=torch.float64))
        with self.assertRaises(ValueError):
            self.alignment.sft_loss(logits, ids, mask)

    def test_completion_mask_rejects_first_position_shape_and_nonbinary(self):
        logits, ids, mask = self.completion_fixture()
        for invalid in (torch.ones_like(mask), mask[:, :-1], torch.full_like(mask, .5, dtype=torch.float64)):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    self.alignment.completion_log_probs(logits, ids, invalid)

    def test_completion_rejects_bad_token_ids(self):
        logits, ids, mask = self.completion_fixture()
        ids[0, 1] = 2
        with self.assertRaises(ValueError):
            self.alignment.completion_log_probs(logits, ids, mask)

    def test_dpo_equal_margins_is_log_two_and_references_detach(self):
        pc = torch.tensor([-2., -3.], requires_grad=True)
        pr = torch.tensor([-2., -3.], requires_grad=True)
        rc = torch.tensor([-4., -5.], requires_grad=True)
        rr = torch.tensor([-4., -5.], requires_grad=True)
        loss = self.alignment.dpo_loss(pc, pr, rc, rr, beta=.5)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue((pc.grad < 0).all().item())
        self.assertTrue((pr.grad > 0).all().item())
        self.assertIsNone(rc.grad)
        self.assertIsNone(rr.grad)

    def test_dpo_better_chosen_margin_lowers_loss_and_is_extreme_stable(self):
        zeros = torch.zeros(1)
        good = self.alignment.dpo_loss(torch.tensor([10000.]), zeros, zeros, zeros, beta=1.)
        bad = self.alignment.dpo_loss(torch.tensor([-10000.]), zeros, zeros, zeros, beta=1.)
        self.assertTrue(torch.isfinite(good).item() and torch.isfinite(bad).item())
        self.assertLess(good.item(), math.log(2))
        self.assertAlmostEqual(bad.item(), 10000.)

    def test_half_precision_dpo_margin_uses_fp32_and_keeps_policy_gradient(self):
        for dtype, expected in ((torch.float16, 7999.8), (torch.bfloat16, 7987.0)):
            with self.subTest(dtype=dtype):
                pc = torch.tensor([-40000.], dtype=dtype, requires_grad=True)
                pr = torch.tensor([-1.], dtype=dtype, requires_grad=True)
                rc = torch.tensor([-1.], dtype=dtype, requires_grad=True)
                rr = torch.tensor([-40000.], dtype=dtype, requires_grad=True)
                loss = self.alignment.dpo_loss(pc, pr, rc, rr, beta=.1)
                self.assertTrue(torch.isfinite(loss).item(), "finite DPO inputs overflowed in the margin")
                self.assertEqual(loss.dtype, torch.float32)
                self.assertAlmostEqual(loss.item(), expected, delta=.001)
                loss.backward()
                self.assertAlmostEqual(pc.grad.item(), -.1, delta=.001)
                self.assertAlmostEqual(pr.grad.item(), .1, delta=.001)
                self.assertIsNone(rc.grad)
                self.assertIsNone(rr.grad)

    def test_dpo_rejects_mismatched_pairs_and_invalid_beta(self):
        for beta in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.alignment.dpo_loss(*[torch.zeros(2) for _ in range(4)], beta=beta)
        with self.assertRaises(ValueError):
            self.alignment.dpo_loss(torch.zeros(2), torch.zeros(3), torch.zeros(2), torch.zeros(2))

    def test_group_advantages_population_std_and_constant_group(self):
        rewards = torch.tensor([[1., 3.], [7., 7.]], requires_grad=True)
        advantages, valid = self.alignment.group_advantages(rewards)
        torch.testing.assert_close(advantages, torch.tensor([[-1., 1.], [0., 0.]]))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertFalse(advantages.requires_grad)

    def test_half_precision_group_centering_avoids_rounded_group_mean(self):
        # Group means 40016 (fp16) and 257 (bf16) are not representable there.
        for dtype, rewards in ((torch.float16, [[40000., 40032.]]), (torch.bfloat16, [[256., 258.]])):
            with self.subTest(dtype=dtype):
                advantages, valid = self.alignment.group_advantages(torch.tensor(rewards, dtype=dtype))
                self.assertEqual(advantages.dtype, torch.float32)
                torch.testing.assert_close(advantages, torch.tensor([[-1., 1.]]))
                self.assertEqual(valid.tolist(), [True])

    def test_group_advantages_all_constant_and_single_sample_are_invalid(self):
        for rewards in (torch.ones(2, 3), torch.tensor([[1.], [2.]])):
            advantages, valid = self.alignment.group_advantages(rewards)
            self.assertEqual(advantages.abs().sum().item(), 0)
            self.assertFalse(valid.any().item())
        with self.assertRaises(ValueError):
            self.alignment.group_advantages(torch.tensor([[float("nan"), 1.]]))

    def test_grpo_clips_positive_and_negative_advantages(self):
        # ratio=2: +1 clipped to 1.2, -1 conservatively uses -2.
        current = torch.full((2, 1), math.log(2), requires_grad=True)
        old = torch.zeros_like(current, requires_grad=True)
        advantages = torch.tensor([1., -1.], requires_grad=True)
        loss = self.alignment.grpo_loss(current, old, advantages, torch.ones_like(current, dtype=torch.bool))
        self.assertAlmostEqual(loss.item(), .4, places=6)
        loss.backward()
        self.assertAlmostEqual(current.grad[0, 0].item(), 0)
        self.assertAlmostEqual(current.grad[1, 0].item(), 1)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)

    def test_grpo_excludes_constant_groups_in_combined_mask(self):
        advantages, valid = self.alignment.group_advantages(torch.tensor([[1., 3.], [5., 5.]]))
        current = torch.tensor([[[0.], [math.log(1.1)]], [[10.], [10.]]], requires_grad=True)
        old = torch.zeros_like(current)
        completion_mask = torch.ones_like(current, dtype=torch.bool)
        mask = completion_mask & valid[:, None, None]
        loss = self.alignment.grpo_loss(current, old, advantages, mask)
        self.assertAlmostEqual(loss.item(), -.05, places=6)
        loss.backward()
        self.assertEqual(current.grad[1].abs().sum().item(), 0)

    def test_grpo_empty_mask_returns_finite_gradient_connected_zero(self):
        current = torch.zeros(2, 3, requires_grad=True)
        loss = self.alignment.grpo_loss(current, torch.zeros_like(current), torch.ones(2), torch.zeros_like(current, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.zeros_like(current))

    def test_grpo_k3_kl_value_and_detached_reference(self):
        current = torch.tensor([[math.log(.5)]], requires_grad=True)
        reference = torch.tensor([[0.]], requires_grad=True)
        loss = self.alignment.grpo_loss(current, current.detach(), torch.zeros(1), torch.ones(1, 1, dtype=torch.bool), ref_log_probs=reference, kl_beta=.5)
        self.assertAlmostEqual(loss.item(), .5 * (2 - math.log(2) - 1), places=6)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertIsNotNone(current.grad)

    def test_half_precision_grpo_probability_math_is_fp32(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                current = torch.tensor([[-1.]], dtype=dtype)
                old = torch.tensor([[-13.]], dtype=dtype)
                mask = torch.ones(1, 1, dtype=torch.bool)
                loss = self.alignment.grpo_loss(current, old, torch.tensor([-1.], dtype=dtype), mask)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertAlmostEqual(loss.item(), math.exp(12), delta=.05)
                # The k3 penalty likewise needs a wider exponential computation.
                current = torch.tensor([[-12.]], dtype=dtype)
                loss = self.alignment.grpo_loss(current, current, torch.zeros(1, dtype=dtype), mask, ref_log_probs=torch.zeros_like(current), kl_beta=1.)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertAlmostEqual(loss.item(), math.exp(12) - 13, delta=.05)

    def test_grpo_rejects_bad_shapes_and_missing_kl_reference(self):
        current = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            self.alignment.grpo_loss(current, current, torch.ones(2), torch.ones(2, 2))
        with self.assertRaises(ValueError):
            self.alignment.grpo_loss(current, current, torch.ones(2), torch.ones_like(current), kl_beta=.1)

    def test_grpo_rejects_active_ratio_overflow_before_nan_gradients(self):
        current = torch.tensor([[1000.]], requires_grad=True)
        with self.assertRaises(ValueError):
            self.alignment.grpo_loss(current, torch.zeros_like(current), torch.ones(1), torch.ones_like(current, dtype=torch.bool))

    def test_cli_reports_only_deterministic_math_diagnostics(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "alignment_check.py"
        completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "math_checks_passed")
        self.assertEqual(report["scope"], "deterministic structural diagnostic; no training or benchmark data")
        self.assertAlmostEqual(report["dpo_equal_margins"], math.log(2), places=6)
        self.assertEqual(report["valid_groups"], [True, False])


if __name__ == "__main__":
    unittest.main()
