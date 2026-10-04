"""Exact small-state checks for educational categorical and mask diffusion."""
import importlib.util
import itertools
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: diffusion numerics were not tested")
class CategoricalDiffusionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from fm_tutorial.diffusion.discrete import CategoricalDiffusion
        self.torch = torch
        self.Diffusion = CategoricalDiffusion
        torch.manual_seed(31)
        self.diffusion = CategoricalDiffusion(3, steps=3, beta_start=0.15, beta_end=0.45)

    def test_row_stochastic_matrices_and_cumulative_product(self):
        torch, diffusion = self.torch, self.diffusion
        identity = torch.eye(3)
        torch.testing.assert_close(diffusion.Q_bar[0], identity)
        torch.testing.assert_close(diffusion.Q[0], identity)
        product = identity
        for t in range(1, 4):
            self.assertTrue((diffusion.Q[t] >= 0).all().item())
            torch.testing.assert_close(diffusion.Q[t].sum(-1), torch.ones(3))
            product = product @ diffusion.Q[t]
            torch.testing.assert_close(diffusion.Q_bar[t], product)
            torch.testing.assert_close(diffusion.Q_bar[t].sum(-1), torch.ones(3))

    def test_forward_marginal_and_clean_identity(self):
        torch = self.torch
        x0 = torch.tensor([[0, 2], [1, 0]])
        t = torch.tensor([1, 3])
        actual = self.diffusion.q_probs(x0, t)
        expected = torch.stack([self.diffusion.Q_bar[1][x0[0]],
                                self.diffusion.Q_bar[3][x0[1]]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(self.diffusion.q_probs(x0, 0),
                                   torch.nn.functional.one_hot(x0, 3).float())
        torch.testing.assert_close(self.diffusion.q_sample(x0, 0), x0)

    def test_full_uniform_noise_endpoint_and_forward_sampling(self):
        torch = self.torch
        diffusion = self.Diffusion(3, steps=1, beta_start=1, beta_end=1)
        clean = torch.zeros((1, 15000), dtype=torch.long)
        torch.testing.assert_close(diffusion.q_probs(clean[:, :2], 1),
                                   torch.full((1, 2, 3), 1 / 3))
        noisy = diffusion.q_sample(clean, 1)
        histogram = torch.bincount(noisy.flatten(), minlength=3).float() / noisy.numel()
        torch.testing.assert_close(histogram, torch.full((3,), 1 / 3), atol=0.015, rtol=0)

    def test_posterior_matches_enumerated_complete_paths(self):
        # Enumerate paths rather than rebuilding the implementation's matrix formula.
        torch, diffusion = self.torch, self.diffusion
        for t, clean, observed in itertools.product(range(1, 4), range(3), range(3)):
            joint = torch.zeros(3, dtype=torch.float64)
            for path in itertools.product(range(3), repeat=t - 1):
                states = (clean,) + path + (observed,)
                probability = 1.0
                for k in range(1, t + 1):
                    probability *= diffusion.Q[k, states[k - 1], states[k]].item()
                joint[states[t - 1]] += probability
            expected = (joint / joint.sum()).float()
            actual = diffusion.posterior_probs(torch.tensor([[clean]]),
                                               torch.tensor([[observed]]), t)[0, 0]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_batched_posteriors_normalize_and_first_step_is_clean(self):
        torch = self.torch
        x0 = torch.tensor([[0, 1, 2], [2, 0, 1]])
        xt = torch.tensor([[2, 2, 1], [0, 1, 1]])
        posterior = self.diffusion.posterior_probs(x0, xt, torch.tensor([1, 3]))
        torch.testing.assert_close(posterior.sum(-1), torch.ones_like(x0).float())
        torch.testing.assert_close(posterior[0], torch.nn.functional.one_hot(x0[0], 3).float())

    def test_reverse_mixes_each_normalized_clean_posterior(self):
        torch = self.torch
        diffusion = self.Diffusion(2, steps=2, beta_start=0.2, beta_end=0.4)
        prediction = torch.tensor([[[0.5, 0.5]]])
        xt = torch.tensor([[0]])
        actual = diffusion.reverse_probs(prediction, xt, 2)
        expected = torch.tensor([[[0.5 * (0.72 / 0.74 + 0.08 / 0.26),
                                   0.5 * (0.02 / 0.74 + 0.18 / 0.26)]]])
        torch.testing.assert_close(actual, expected)
        # Normalizing after mixing joints would incorrectly produce [.8, .2].
        self.assertGreater(abs(actual[0, 0, 0].item() - 0.8), 0.1)
        torch.testing.assert_close(diffusion.reverse_probs(prediction, xt, 1), prediction)

    def test_reverse_matches_brute_clean_mixture_and_keeps_gradients(self):
        torch = self.torch
        logits = torch.tensor([[[0.1, -0.4, 0.7], [0.2, 0.5, -0.3]]], requires_grad=True)
        clean_probs = logits.softmax(-1)
        xt = torch.tensor([[1, 2]])
        actual = self.diffusion.reverse_probs(clean_probs, xt, 3)
        expected = torch.zeros_like(actual)
        for clean in range(3):
            conditional = self.diffusion.posterior_probs(torch.full_like(xt, clean), xt, 3)
            expected = expected + clean_probs[..., clean, None] * conditional
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.ones((1, 2)))
        (-actual[..., 0].log().mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_sampler_passes_integer_times_and_reaches_model_clean_prediction(self):
        torch = self.torch
        calls = []
        def model(ids, t):
            calls.append((ids.clone(), t.clone()))
            logits = torch.full((*ids.shape, 3), -1000.0)
            logits[..., 2] = 1000.0
            return logits
        result = self.diffusion.sample(model, (2, 4))
        torch.testing.assert_close(result, torch.full((2, 4), 2))
        self.assertEqual([t[0].item() for _, t in calls], [3, 2, 1])
        self.assertTrue(all(t.dtype == torch.long and tuple(t.shape) == (2,) for _, t in calls))

    def test_invalid_configuration_times_and_probabilities_are_rejected(self):
        torch = self.torch
        for kwargs in ({"vocab_size": 1}, {"vocab_size": 3, "steps": 0},
                       {"vocab_size": 3, "beta_start": 0},
                       {"vocab_size": 3, "beta_start": 1e-46},
                       {"vocab_size": 3, "beta_end": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.Diffusion(**kwargs)
        ids = torch.tensor([[0, 1]])
        for t in (-1, 4, torch.tensor([1.5]), torch.tensor([1, 2])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                self.diffusion.q_probs(ids, t)
        with self.assertRaises(ValueError):
            self.diffusion.posterior_probs(ids, ids, 0)
        with self.assertRaises(ValueError):
            self.diffusion.reverse_probs(torch.ones((1, 2, 3)), ids, 2)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: diffusion numerics were not tested")
class MaskedDiffusionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from fm_tutorial.diffusion.discrete import mask_tokens, masked_loss, sample_masked, MaskedDenoiser
        self.torch = torch
        self.mask_tokens, self.masked_loss = mask_tokens, masked_loss
        self.sample_masked, self.Model = sample_masked, MaskedDenoiser
        torch.manual_seed(37)

    def test_corruption_endpoints_and_immutable_conditions(self):
        torch = self.torch
        clean = torch.tensor([[1, 2, 3], [3, 2, 1]])
        conditions = torch.tensor([[True, False, False], [False, True, False]])
        noisy, masked = self.mask_tokens(clean, torch.tensor([0.0, 1.0]), 0, conditions)
        torch.testing.assert_close(noisy[0], clean[0])
        torch.testing.assert_close(noisy[1], torch.tensor([0, 2, 0]))
        torch.testing.assert_close(masked, torch.tensor([[False, False, False], [True, False, True]]))
        torch.testing.assert_close(clean, torch.tensor([[1, 2, 3], [3, 2, 1]]))

    def test_corruption_probability_tracks_linear_time(self):
        torch = self.torch
        clean = torch.ones((2, 20000), dtype=torch.long)
        _, masked = self.mask_tokens(clean, torch.tensor([0.2, 0.8]), 0)
        torch.testing.assert_close(masked.float().mean(-1), torch.tensor([0.2, 0.8]),
                                   atol=0.015, rtol=0)

    def test_masked_loss_is_float32_ce_only_on_masked_positions(self):
        torch = self.torch
        logits = torch.tensor([[[2.0, -1.0, 0.5], [0.1, 2.0, -0.5]]],
                              dtype=torch.float16, requires_grad=True)
        targets = torch.tensor([[2, 1]])
        masked = torch.tensor([[True, False]])
        loss = self.masked_loss(logits, targets, masked)
        expected = torch.nn.functional.cross_entropy(logits[:, :1].float().reshape(-1, 3), targets[:, :1].reshape(-1))
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad[0, 0].abs().sum().item(), 0)
        torch.testing.assert_close(logits.grad[0, 1], torch.zeros(3, dtype=torch.float16))

    def test_all_visible_loss_is_finite_differentiable_zero(self):
        torch = self.torch
        # An empty reduction must avoid overflow or -inf * 0 in excluded logits.
        logits = torch.full((2, 3, 4), 60000.0, dtype=torch.float16, requires_grad=True)
        loss = self.masked_loss(logits, torch.ones((2, 3), dtype=torch.long),
                                torch.zeros((2, 3), dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        excluded_infinity = torch.full((1, 2, 3), -torch.inf, requires_grad=True)
        zero = self.masked_loss(excluded_infinity, torch.ones((1, 2), dtype=torch.long),
                                torch.zeros((1, 2), dtype=torch.bool))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        torch.testing.assert_close(excluded_infinity.grad, torch.zeros_like(excluded_infinity))

    def test_all_mask_sampling_has_no_target_access_and_excludes_mask_id(self):
        torch = self.torch
        seen = []
        def model(ids, t):
            seen.append((ids.clone(), t.clone()))
            logits = torch.full((*ids.shape, 6), -1000.0)
            logits[..., 2] = 2000.0  # The mask is deliberately most likely.
            logits[..., 5] = 1000.0
            return logits
        result = self.sample_masked(model, (2, 7), mask_id=2, steps=4)
        torch.testing.assert_close(seen[0][0], torch.full((2, 7), 2))
        torch.testing.assert_close(seen[0][1], torch.ones(2))
        torch.testing.assert_close(result, torch.full((2, 7), 5))
        for (before, _), (after, _) in zip(seen, seen[1:]):
            torch.testing.assert_close(after[before != 2], before[before != 2])
        self.assertTrue(all(t.dtype == torch.float32 for _, t in seen))

    def test_sampler_preserves_prompt_and_ignores_unconditioned_ids(self):
        torch = self.torch
        conditions = torch.tensor([[True, False, False, True], [False, True, False, False]])
        ids = torch.tensor([[1, 99, 98, 4], [97, 3, 96, 95]])
        seen = []
        def model(noisy, t):
            seen.append(noisy.clone())
            logits = torch.full((*noisy.shape, 7), -1000.0)
            logits[..., 6] = 1000.0
            return logits
        result = self.sample_masked(model, (2, 4), mask_id=2, steps=3,
                                    condition_ids=ids, condition_mask=conditions)
        torch.testing.assert_close(seen[0][~conditions], torch.full((5,), 2))
        for noisy in seen:
            torch.testing.assert_close(noisy[conditions], ids[conditions])
        torch.testing.assert_close(result[conditions], ids[conditions])
        torch.testing.assert_close(result[~conditions], torch.full((5,), 6))

    def test_reveal_probability_is_exact_absorbing_conditional(self):
        torch = self.torch
        seen = []
        def model(ids, t):
            seen.append(ids.clone())
            return torch.zeros((*ids.shape, 3))
        result = self.sample_masked(model, (1, 20000), mask_id=1, steps=4)
        # At times 1, .75, .5, .25 the mask marginal must equal time.
        self.assertEqual(len(seen), 4)
        for k, ids in enumerate(seen):
            self.assertAlmostEqual((ids == 1).float().mean().item(), 1 - k / 4, delta=0.02)
        self.assertFalse((result == 1).any().item())

    def test_single_step_and_fully_conditioned_sampling(self):
        torch = self.torch
        def model(ids, t):
            return torch.zeros((*ids.shape, 5))
        result = self.sample_masked(model, (1, 3), mask_id=3, steps=1)
        self.assertFalse((result == 3).any().item())
        conditions = torch.tensor([[1, 2, 4]])
        result = self.sample_masked(model, (1, 3), mask_id=3, steps=2,
                                    condition_ids=conditions, condition_mask=torch.ones((1, 3), dtype=torch.bool))
        torch.testing.assert_close(result, conditions)

    def test_denoiser_is_bidirectional_and_time_conditioned(self):
        torch = self.torch
        model = self.Model(8, max_length=6, dim=16, heads=4, layers=2).eval()
        a = torch.tensor([[1, 2, 3, 4, 5, 6]])
        b = torch.tensor([[1, 2, 3, 7, 7, 7]])
        with torch.no_grad():
            left, right = model(a, torch.tensor([0.5])), model(b, torch.tensor([0.5]))
            other_time = model(a, torch.tensor([1.0]))
        self.assertEqual(tuple(left.shape), (1, 6, 8))
        self.assertGreater((left[:, :3] - right[:, :3]).abs().max().item(), 1e-4)
        self.assertGreater((left - other_time).abs().max().item(), 1e-4)

    def test_actual_masked_loss_gradients_reach_embedding_attention_and_time(self):
        torch = self.torch
        model = self.Model(8, max_length=5, dim=16, heads=4, layers=1)
        clean = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        noisy, masked = self.mask_tokens(clean, torch.ones(2), mask_id=0)
        loss = self.masked_loss(model(noisy, torch.ones(2)), clean, masked)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        for parameter in (model.token_embedding.weight,
                          model.encoder.layers[0].self_attn.in_proj_weight,
                          model.time_embedding[0].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_invalid_masking_sampling_and_model_dimensions_are_rejected(self):
        torch = self.torch
        clean = torch.tensor([[1, 2]])
        with self.assertRaises(ValueError):
            self.mask_tokens(clean, torch.tensor([1.1]), mask_id=0)
        with self.assertRaises(ValueError):
            self.mask_tokens(clean, torch.tensor([1.0]), mask_id=1)
        with self.assertRaises(ValueError):
            self.sample_masked(lambda x, t: torch.zeros((*x.shape, 3)), (1, 2), 0,
                               condition_ids=clean)
        with self.assertRaises(ValueError):
            self.sample_masked(lambda x, t: torch.zeros((*x.shape, 3)), (1, 2), 4)
        with self.assertRaises(ValueError):
            self.Model(4, max_length=3, dim=15, heads=4)
        model = self.Model(4, max_length=3)
        with self.assertRaises(ValueError):
            model(torch.ones((1, 4), dtype=torch.long), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
