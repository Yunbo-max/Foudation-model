"""Equation and endpoint fixtures; small shapes do not measure generation quality."""
import importlib.util
import math
import unittest
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: diffusion numerics were not tested")
class GaussianDiffusionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from fm_tutorial.diffusion.continuous import GaussianDiffusion
        self.torch = torch
        self.Diffusion = GaussianDiffusion
        torch.manual_seed(29)
        self.diffusion = GaussianDiffusion(steps=20, schedule="linear")

    def oracle(self, x0):
        torch, diffusion = self.torch, self.diffusion
        def epsilon(xt, t):
            a = diffusion.alpha_bar[t].to(xt).reshape((-1,) + (1,) * (xt.ndim - 1))
            return (xt - a.sqrt() * x0) / (1 - a).sqrt()
        return epsilon

    def test_linear_and_cosine_schedules_follow_their_definitions(self):
        torch = self.torch
        expected = torch.linspace(1e-4, 0.02, 20)
        torch.testing.assert_close(self.diffusion.betas, expected)
        torch.testing.assert_close(self.diffusion.alpha_bar, (1 - expected).cumprod(0))
        cosine = self.Diffusion(steps=20, schedule="cosine")
        # Independently compute adjacent cosine ratios in Python scalar math.
        f = [math.cos((i / 20 + 0.008) / 1.008 * math.pi / 2) ** 2 for i in range(21)]
        expected = torch.tensor([min(1 - f[i + 1] / f[i], 0.999) for i in range(20)])
        torch.testing.assert_close(cosine.betas, expected)
        self.assertTrue((cosine.alpha_bar[1:] < cosine.alpha_bar[:-1]).all().item())
        self.assertGreater(cosine.alpha_bar[-1].item(), 0)
        self.assertLess(cosine.alpha_bar[-1].item(), 1e-4)

    def test_forward_corruption_and_oracle_x0_reconstruction(self):
        torch = self.torch
        x0, noise = torch.randn(20, 3), torch.randn(20, 3)
        t = torch.arange(20)
        a = self.diffusion.alpha_bar[:, None]
        xt = self.diffusion.q_sample(x0, t, noise)
        torch.testing.assert_close(xt, a.sqrt() * x0 + (1 - a).sqrt() * noise)
        torch.testing.assert_close(self.diffusion.predict_x0(xt, t, noise), x0)

    def test_posterior_matches_conditional_gaussian_reference(self):
        torch = self.torch
        x0, xt = torch.randn(3, 2), torch.randn(3, 2)
        t = torch.tensor([0, 4, 19])
        mean, variance = self.diffusion.posterior(x0, xt, t)
        self.assertEqual(tuple(variance.shape), (3, 1))
        torch.testing.assert_close(mean[0], x0[0], rtol=0, atol=0)
        self.assertEqual(variance[0].item(), 0)
        for row, index in enumerate(t.tolist()[1:], 1):
            # Multiply q(x_previous|x0) and q(xt|x_previous) via precisions.
            alpha = 1 - self.diffusion.betas[index]
            prior_variance = 1 - self.diffusion.alpha_bar[index - 1]
            likelihood_variance = self.diffusion.betas[index]
            reference_variance = 1 / (1 / prior_variance + alpha / likelihood_variance)
            reference_mean = reference_variance * (
                self.diffusion.alpha_bar[index - 1].sqrt() * x0[row] / prior_variance
                + alpha.sqrt() * xt[row] / likelihood_variance)
            torch.testing.assert_close(variance[row, 0], reference_variance)
            torch.testing.assert_close(mean[row], reference_mean, rtol=2e-5, atol=2e-5)

    def test_ddpm_final_step_has_no_noise_and_mixed_batch_is_masked(self):
        torch = self.torch
        x0 = torch.randn(2, 3)
        t = torch.zeros(2, dtype=torch.long)
        xt = self.diffusion.q_sample(x0, t)
        with patch("torch.randn_like", side_effect=AssertionError("final step drew noise")):
            sample = self.diffusion.p_sample(self.oracle(x0), xt, t)
        torch.testing.assert_close(sample, x0)
        t = torch.tensor([0, 8])
        mean, variance = self.diffusion.posterior(x0, xt, t)
        noise = torch.full_like(xt, 50)
        sample = self.diffusion.p_sample(self.oracle(x0), xt, t, noise=noise)
        torch.testing.assert_close(sample[0], x0[0])
        torch.testing.assert_close(sample, mean + variance.sqrt() * noise)

    def test_ddim_skipping_uses_selected_previous_time(self):
        torch = self.torch
        x0, noise = torch.randn(2, 3), torch.randn(2, 3)
        t, previous = torch.tensor([19, 15]), torch.tensor([9, 3])
        xt = self.diffusion.q_sample(x0, t, noise)
        extra_noise, eta = torch.randn_like(xt), 0.6
        a, b = self.diffusion.alpha_bar[t, None], self.diffusion.alpha_bar[previous, None]
        sigma = eta * (((1 - b) / (1 - a)) * (1 - a / b)).sqrt()
        reference = b.sqrt() * x0 + (1 - b - sigma.square()).sqrt() * noise + sigma * extra_noise
        actual = self.diffusion.ddim_step(self.oracle(x0), xt, t, previous, eta=eta, noise=extra_noise)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
        mixed = self.diffusion.ddim_step(self.oracle(x0), xt, t, torch.tensor([9, -1]),
                                         eta=eta, noise=extra_noise)
        torch.testing.assert_close(mixed[0], reference[0], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(mixed[1], x0[1], atol=2e-5, rtol=2e-5)

    def test_ddim_eta_zero_is_deterministic_and_clean_endpoint_is_x0(self):
        torch = self.torch
        x0 = torch.randn(2, 3)
        t = torch.tensor([19, 10])
        xt = self.diffusion.q_sample(x0, t)
        previous = torch.tensor([5, -1])
        left = self.diffusion.ddim_step(self.oracle(x0), xt, t, previous, noise=torch.zeros_like(xt))
        right = self.diffusion.ddim_step(self.oracle(x0), xt, t, previous, noise=torch.full_like(xt, 99))
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        torch.testing.assert_close(left[1], x0[1])
        with patch("torch.randn_like", side_effect=AssertionError("deterministic DDIM drew noise")):
            without_noise = self.diffusion.ddim_step(self.oracle(x0), xt, t, previous)
        torch.testing.assert_close(without_noise, left, atol=0, rtol=0)
        final = self.diffusion.ddim_step(self.oracle(x0), xt, t, torch.full_like(t, -1),
                                         eta=1, noise=torch.full_like(xt, 99))
        torch.testing.assert_close(final, x0)
        with patch("torch.randn_like", side_effect=AssertionError("clean DDIM endpoint drew noise")):
            final = self.diffusion.ddim_step(self.oracle(x0), xt, t, torch.full_like(t, -1), eta=1)
        torch.testing.assert_close(final, x0)

    def test_ddim_eta_one_adjacent_step_matches_ddpm(self):
        torch = self.torch
        x0 = torch.randn(3, 4)
        t = torch.tensor([0, 6, 19])
        xt, noise = self.diffusion.q_sample(x0, t), torch.randn_like(x0)
        ddpm = self.diffusion.p_sample(self.oracle(x0), xt, t, noise)
        ddim = self.diffusion.ddim_step(self.oracle(x0), xt, t, t - 1, eta=1, noise=noise)
        torch.testing.assert_close(ddim, ddpm, rtol=2e-5, atol=2e-5)

    def test_sampling_steps_and_oracle_clean_endpoint(self):
        torch = self.torch
        x0 = torch.randn(2, 3)
        times = []
        def model(xt, t):
            times.append(t[0].item())
            self.assertEqual(t.dtype, torch.long)
            return self.oracle(x0)(xt, t)
        torch.manual_seed(91)
        left = self.diffusion.sample(model, x0.shape, sampler="ddim", sampling_steps=4)
        self.assertEqual(len(times), 4)
        self.assertEqual(times[0], 19)
        self.assertEqual(times[-1], 0)
        self.assertTrue(all(a > b for a, b in zip(times, times[1:])))
        torch.testing.assert_close(left, x0, atol=2e-5, rtol=2e-5)
        torch.manual_seed(91)
        right = self.diffusion.sample(self.oracle(x0), x0.shape, sampler="ddim", sampling_steps=4)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        one = self.diffusion.sample(self.oracle(x0), x0.shape, sampler="ddim", sampling_steps=1)
        torch.testing.assert_close(one, x0, atol=2e-5, rtol=2e-5)

    def test_arbitrary_shapes_and_training_gradients(self):
        from fm_tutorial.diffusion.models import TimeMLP
        torch = self.torch
        # Tabular, image, video, and time-dependent voxel-field layouts.
        for shape in ((2, 4), (2, 1, 3, 3), (2, 2, 1, 3, 3), (2, 2, 1, 2, 2, 2)):
            with self.subTest(shape=shape):
                x0, noise = torch.randn(shape), torch.randn(shape)
                t = torch.tensor([0, 19])
                xt = self.diffusion.q_sample(x0, t, noise)
                torch.testing.assert_close(self.diffusion.predict_x0(xt, t, noise), x0)
                model = TimeMLP(math.prod(shape[1:]), hidden_dim=16)
                prediction = model(xt, (t.float() + 1) / self.diffusion.steps)
                self.assertEqual(tuple(prediction.shape), shape)
                (prediction - noise).square().mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)
                wrapper = lambda x, time: model(x, (time.float() + 1) / self.diffusion.steps)
                for sampler in ("ddpm", "ddim"):
                    generated = self.diffusion.sample(wrapper, shape, sampler=sampler)
                    self.assertEqual(tuple(generated.shape), shape)
                    self.assertTrue(torch.isfinite(generated).all().item())
                    self.assertFalse(generated.requires_grad)

    def test_invalid_schedules_times_and_shapes_are_rejected(self):
        torch = self.torch
        for kwargs in ({"steps": 0}, {"steps": 1.5}, {"schedule": "bad"},
                       {"beta_start": 0}, {"beta_end": 1}, {"beta_start": 0.1, "beta_end": 0.01},
                       {"schedule": "linear", "beta_start": 1e-10, "beta_end": 1e-9},
                       {"schedule": "linear", "steps": 100, "beta_start": 0.99, "beta_end": 0.999}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.Diffusion(**kwargs)
        x = torch.randn(2, 3)
        for t in (torch.tensor([-1, 0]), torch.tensor([0, 20]), torch.tensor([0., 1.]), torch.tensor([0])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                self.diffusion.q_sample(x, t)
        with self.assertRaises(ValueError):
            self.diffusion.q_sample(x, torch.tensor([0, 1]), noise=torch.zeros(2, 1))
        for kwargs in ({"sampler": "bad"}, {"sampler": "ddpm", "sampling_steps": 3},
                       {"sampler": "ddim", "sampling_steps": 21}, {"sampler": "ddim", "eta": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.diffusion.sample(lambda x, t: x, x.shape, **kwargs)
        with self.assertRaises(ValueError):
            self.diffusion.ddim_step(lambda x, t: x, x, torch.tensor([4, 5]), torch.tensor([4, 3]))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: VP numerics were not tested")
class VPSDETests(unittest.TestCase):
    def setUp(self):
        import torch
        from fm_tutorial.diffusion.continuous import VPSDE
        self.torch, self.VP = torch, VPSDE
        self.sde = VPSDE(beta_min=0.1, beta_max=20)
        torch.manual_seed(29)

    def test_analytical_marginal_and_epsilon_score(self):
        torch = self.torch
        x0, noise = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 2, dtype=torch.float64)
        t = torch.tensor([0.001, 0.3, 1.0], dtype=torch.float64)
        integrated_beta = 0.1 * t + 0.5 * 19.9 * t.square()
        coefficient = torch.exp(-0.5 * integrated_beta)[:, None]
        std = (1 - torch.exp(-integrated_beta)).sqrt()[:, None]
        mean, actual_std = self.sde.marginal_stats(x0, t)
        xt = self.sde.marginal(x0, t, noise)
        torch.testing.assert_close(mean, coefficient * x0)
        torch.testing.assert_close(actual_std, std)
        torch.testing.assert_close(xt, mean + std * noise)
        conditional_score = -(xt - mean) / std.square()
        torch.testing.assert_close(self.sde.score_from_epsilon(noise, t), conditional_score)

    def test_reverse_sde_negative_dt_and_ode_half_score_reference(self):
        torch = self.torch
        xt, score, noise = torch.tensor([[2., -1.], [1., 3.]]), torch.tensor([[0.3, -0.7], [0.2, 0.5]]), torch.ones(2, 2)
        t, dt = torch.tensor([0.8, 0.3]), -0.01
        beta = (0.1 + 19.9 * t)[:, None]
        reference = xt + (-0.5 * beta * xt - beta * score) * dt + (-beta * dt).sqrt() * noise
        actual = self.sde.reverse_step(lambda x, time: score, xt, t, dt, noise=noise)
        torch.testing.assert_close(actual, reference)
        reference_ode = xt + (-0.5 * beta * xt - 0.5 * beta * score) * dt
        with patch("torch.randn_like", side_effect=AssertionError("ODE drew noise")):
            actual_ode = self.sde.reverse_step(lambda x, time: score, xt, t, dt, probability_flow=True)
        torch.testing.assert_close(actual_ode, reference_ode)

    def test_near_clean_float32_standard_deviation_avoids_cancellation(self):
        torch = self.torch
        x0, t = torch.ones(1, 2), torch.tensor([1e-8])
        _, std = self.sde.marginal_stats(x0, t)
        self.assertGreater(std.item(), 0)
        # First-order variance beta_min*t is accurate at this tiny time.
        reference = torch.tensor([[math.sqrt(0.1 * 1e-8)]])
        torch.testing.assert_close(std, reference, rtol=2e-6, atol=0)
        score = self.sde.score_from_epsilon(torch.ones_like(x0), t)
        self.assertTrue(torch.isfinite(score).all().item())
        torch.testing.assert_close(score, -torch.ones_like(x0) / reference, rtol=2e-6, atol=0)

    def test_stationary_gaussian_score_leaves_probability_flow_unchanged(self):
        torch = self.torch
        # N(0,I) is stationary under VP: f - (g^2 / 2)*(-x) = 0.
        torch.manual_seed(73)
        initial = torch.randn(2, 3)
        torch.manual_seed(73)
        sampled = self.sde.sample(lambda x, t: -x, initial.shape, steps=7, probability_flow=True)
        torch.testing.assert_close(sampled, initial, rtol=0, atol=0)

    def test_sampling_uses_descending_times_and_preserves_shapes(self):
        torch = self.torch
        for shape in ((2, 4), (2, 1, 3, 3), (2, 2, 1, 3, 3), (2, 2, 1, 2, 2, 2)):
            for probability_flow in (False, True):
                times = []
                def score(x, t):
                    times.append(t[0].item())
                    return -x
                sample = self.sde.sample(score, shape, steps=8, t_min=0.01, probability_flow=probability_flow)
                self.assertEqual(tuple(sample.shape), shape)
                self.assertTrue(torch.isfinite(sample).all().item())
                self.assertEqual(len(times), 8)
                self.assertEqual(times[0], 1)
                self.assertGreater(times[-1], 0.01)
                self.assertTrue(all(a > b for a, b in zip(times, times[1:])))

    def test_marginal_and_score_remain_differentiable(self):
        torch = self.torch
        x0, eps = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        t = torch.tensor([0.2, 0.8], requires_grad=True)
        xt = self.sde.marginal(x0, t, eps)
        (xt.square().mean() + self.sde.score_from_epsilon(eps, t).square().mean()).backward()
        for value in (x0, eps, t):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all().item())

    def test_invalid_parameters_and_reverse_steps_are_rejected(self):
        torch = self.torch
        for kwargs in ({"beta_min": 0}, {"beta_max": -1}, {"beta_min": 3, "beta_max": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.VP(**kwargs)
        x = torch.randn(2, 3)
        for t in (torch.tensor([0., 0.2]), torch.tensor([0.2, 1.1]), torch.tensor([0, 1]), torch.tensor([0.2])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                self.sde.marginal(x, t)
        with self.assertRaises(ValueError):
            self.sde.reverse_step(lambda x, t: -x, x, torch.tensor([0.2, 0.3]), 0.01)
        with self.assertRaises(ValueError):
            self.sde.reverse_step(lambda x, t: -x, x, torch.tensor([0.2, 0.3]), -0.4)
        for kwargs in ({"steps": 0}, {"t_min": 0}, {"t_min": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sde.sample(lambda x, t: -x, x.shape, **kwargs)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: time network was not tested")
class TimeMLPTests(unittest.TestCase):
    def test_time_validation_and_time_influence(self):
        import torch
        from fm_tutorial.diffusion.models import TimeMLP
        torch.manual_seed(29)
        model = TimeMLP(3, hidden_dim=8)
        x = torch.randn(2, 3)
        self.assertFalse(torch.allclose(model(x, torch.zeros(2)), model(x, torch.ones(2))))
        for t in (torch.tensor([-0.1, 1.]), torch.tensor([0., 1.1]), torch.tensor([0, 1]), torch.zeros(1)):
            with self.subTest(t=t), self.assertRaises(ValueError):
                model(x, t)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4), torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
