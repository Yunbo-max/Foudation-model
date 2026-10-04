"""Small numerical fixtures for conditioning; these do not assess sample quality."""
import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: conditioning was not tested")
class ConditioningTests(unittest.TestCase):
    def setUp(self):
        import torch
        from fm_tutorial.diffusion import conditioning
        self.torch, self.conditioning = torch, conditioning
        torch.manual_seed(29)

    def uniform_attention(self, attention):
        """Set Q=K=0, V=identity: the observable result is a token mean."""
        torch = self.torch
        width = attention.embed_dim
        with torch.no_grad():
            attention.in_proj_weight.zero_()
            attention.in_proj_weight[2 * width:].copy_(torch.eye(width))
            attention.in_proj_bias.zero_()
            attention.out_proj.weight.copy_(torch.eye(width))
            attention.out_proj.bias.zero_()

    def test_concat_projects_features_and_broadcast_condition(self):
        torch = self.torch
        layer = self.conditioning.FeatureConcat(2, 1)
        with torch.no_grad():
            layer.proj.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 1., -2.]]))
            layer.proj.bias.copy_(torch.tensor([4., -3.]))
        h = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        c = torch.tensor([[10.], [20.]])
        expected = torch.tensor([[[39., -22.], [45., -22.]],
                                 [[81., -42.], [87., -42.]]])
        torch.testing.assert_close(layer(h, c), expected, atol=0, rtol=0)

    def test_linear_concat_equals_separate_projections_and_one_bias(self):
        torch = self.torch
        layer = self.conditioning.FeatureConcat(4, 3)
        h, c = torch.randn(2, 5, 4), torch.randn(2, 3)
        weights, bias = layer.proj.weight, layer.proj.bias
        expected = h @ weights[:, :4].T + (c @ weights[:, 4:].T)[:, None, :] + bias
        torch.testing.assert_close(layer(h, c), expected)

    def test_add_condition_retains_features_and_broadcasts_to_all_tokens(self):
        torch = self.torch
        layer = self.conditioning.AddCondition(2, 1)
        with torch.no_grad():
            layer.proj.weight.copy_(torch.tensor([[2.], [-1.]]))
            layer.proj.bias.copy_(torch.tensor([1., 3.]))
        h = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        c = torch.tensor([[2.], [-1.]])
        expected = torch.tensor([[[6., 3.], [8., 5.]], [[4., 10.], [6., 12.]]])
        torch.testing.assert_close(layer(h, c), expected, atol=0, rtol=0)

    def test_film_uses_one_plus_scale_and_a_shift_without_normalization(self):
        torch = self.torch
        layer = self.conditioning.FiLM(2, 1)
        with torch.no_grad():
            layer.modulation.weight.copy_(torch.tensor([[1.], [-1.], [2.], [3.]]))
            layer.modulation.bias.zero_()
        h, c = torch.tensor([[[1., 2.], [3., 4.]]]), torch.tensor([[2.]])
        torch.testing.assert_close(layer(h, c), torch.tensor([[[7., 4.], [13., 2.]]]),
                                   atol=0, rtol=0)

    def test_fresh_film_and_adaln_respond_to_conditions(self):
        torch = self.torch
        h = torch.randn(2, 3, 4)
        for cls in (self.conditioning.FiLM, self.conditioning.AdaLN):
            with self.subTest(method=cls.__name__):
                layer = cls(4, 3)
                self.assertFalse(torch.allclose(layer(h, torch.zeros(2, 3)),
                                                 layer(h, torch.ones(2, 3))))

    def test_adaln_normalizes_each_token_before_modulation(self):
        torch = self.torch
        layer = self.conditioning.AdaLN(2, 1).double()
        with torch.no_grad():
            layer.modulation.weight.copy_(torch.tensor([[1.], [-1.], [2.], [3.]]))
            layer.modulation.bias.zero_()
        h = torch.tensor([[[1., 3.], [-2., 0.]]], dtype=torch.float64)
        c = torch.tensor([[2.]], dtype=torch.float64)
        normalized_magnitude = 1 / (1 + 1e-6) ** 0.5
        expected = torch.tensor([[[4 - 3 * normalized_magnitude, 6 - normalized_magnitude],
                                 [4 - 3 * normalized_magnitude, 6 - normalized_magnitude]]],
                                dtype=torch.float64)
        torch.testing.assert_close(layer(h, c), expected)

    def test_adaln_is_approximately_invariant_to_positive_affine_input_changes(self):
        torch = self.torch
        layer = self.conditioning.AdaLN(4, 3).double()
        h = torch.tensor([[[1., -2., 3., 4.], [2., 5., -3., 8.]]], dtype=torch.float64)
        c = torch.randn(1, 3, dtype=torch.float64)
        transformed = h * torch.tensor([[[2.], [3.]]]) + torch.tensor([[[7.], [-5.]]])
        torch.testing.assert_close(layer(h, c), layer(transformed, c), rtol=1e-5, atol=1e-6)

    def test_adaln_zero_is_an_exact_identity_at_initialization(self):
        torch = self.torch
        layer = self.conditioning.AdaLNZeroBlock(8, 3, heads=2)
        h, c = torch.randn(2, 5, 8), torch.randn(2, 3)
        torch.testing.assert_close(layer(h, c), h, rtol=0, atol=0)

    def test_adaln_zero_initially_learns_gates_then_backbone(self):
        torch = self.torch
        layer = self.conditioning.AdaLNZeroBlock(4, 3, heads=2)
        h = torch.randn(2, 3, 4, requires_grad=True)
        c = torch.randn(2, 3)
        layer(h, c).square().mean().backward()
        head = layer.modulation[-1]
        gate_rows = torch.cat((head.weight.grad[8:12], head.weight.grad[20:24]))
        self.assertGreater(gate_rows.abs().sum().item(), 0)
        for start in (0, 4, 12, 16):
            self.assertEqual(head.weight.grad[start:start + 4].abs().sum().item(), 0)
        for parameter in (*layer.attn.parameters(), *layer.mlp.parameters()):
            self.assertEqual(parameter.grad.abs().sum().item(), 0)
        torch.testing.assert_close(h.grad, 2 * h / h.numel())
        torch.optim.SGD(layer.parameters(), lr=0.1).step()
        layer.zero_grad(set_to_none=True)
        layer(h, c).square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.attn.parameters()), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.mlp.parameters()), 0)

    def test_adaln_zero_residual_gates_can_be_negative_or_above_one(self):
        torch = self.torch
        layer = self.conditioning.AdaLNZeroBlock(2, 1, heads=1)
        with torch.no_grad():
            layer.modulation[-1].bias[10:].copy_(torch.tensor([-2., 1.5]))
            layer.mlp[-1].weight.zero_()
            layer.mlp[-1].bias.copy_(torch.tensor([3., 4.]))
        h = torch.tensor([[[1., 2.], [3., 4.]]])
        expected = torch.tensor([[[-5., 8.], [-3., 10.]]])
        torch.testing.assert_close(layer(h, torch.ones(1, 1)), expected, atol=0, rtol=0)

    def test_cross_attention_reads_condition_values_and_preserves_a_residual(self):
        torch = self.torch
        layer = self.conditioning.CrossAttentionCondition(2, 2, heads=1)
        self.uniform_attention(layer.attn)
        h = torch.tensor([[[10., 20.], [30., 40.]]])
        c = torch.tensor([[[1., 3.], [3., 5.]]])
        expected = torch.tensor([[[12., 24.], [32., 44.]]])
        torch.testing.assert_close(layer(h, c), expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(layer(h, c), layer(h, c + 1)))

    def test_cross_attention_masks_padding_values_and_their_gradients(self):
        torch = self.torch
        layer = self.conditioning.CrossAttentionCondition(2, 2, heads=1)
        self.uniform_attention(layer.attn)
        h = torch.tensor([[[10., 20.]]])
        c = torch.tensor([[[1., 3.], [1e6, -1e6]]], requires_grad=True)
        mask = torch.tensor([[False, True]])
        actual = layer(h, c, padding_mask=mask)
        torch.testing.assert_close(actual, torch.tensor([[[11., 23.]]]), atol=0, rtol=0)
        changed_padding = c.detach().clone()
        changed_padding[:, 1] = torch.tensor([-3e6, 5e6])
        torch.testing.assert_close(actual, layer(h, changed_padding, padding_mask=mask),
                                   atol=0, rtol=0)
        actual.sum().backward()
        torch.testing.assert_close(c.grad, torch.tensor([[[1., 1.], [0., 0.]]]),
                                   atol=0, rtol=0)

    def test_attention_accepts_condition_width_different_from_data_width(self):
        torch = self.torch
        h, c = torch.randn(2, 5, 8), torch.randn(2, 3, 6, requires_grad=True)
        for cls in (self.conditioning.CrossAttentionCondition, self.conditioning.PrefixCondition):
            with self.subTest(method=cls.__name__):
                layer = cls(8, 6, heads=2)
                output = layer(h, c)
                self.assertEqual(tuple(output.shape), (2, 5, 8))
                output.square().mean().backward()
                self.assertGreater(c.grad.abs().sum().item(), 0)
                c.grad = None

    def test_prefix_attention_mixes_condition_and_data_bidirectionally(self):
        torch = self.torch
        layer = self.conditioning.PrefixCondition(2, 2, heads=1)
        with torch.no_grad():
            layer.proj.weight.copy_(torch.eye(2))
            layer.proj.bias.zero_()
        self.uniform_attention(layer.attn)
        h = torch.tensor([[[2., 4.], [4., 8.]]])
        c = torch.tensor([[[9., 18.]]])
        expected = torch.tensor([[[7., 14.], [9., 18.]]])
        torch.testing.assert_close(layer(h, c), expected, atol=0, rtol=0)
        changed_future = h.clone()
        changed_future[:, 1] += 3
        self.assertFalse(torch.equal(layer(h, c)[:, 0], layer(changed_future, c)[:, 0]))
        self.assertFalse(torch.equal(layer(h, c), layer(h, c + 1)))

    def test_zero_spatial_residual_preserves_features_for_any_hint_initially(self):
        torch = self.torch
        layer = self.conditioning.ZeroSpatialResidual(2, 3)
        h, hint = torch.randn(2, 2, 3, 4), torch.randn(2, 3, 3, 4)
        torch.testing.assert_close(layer(h, hint), h, atol=0, rtol=0)
        torch.testing.assert_close(layer(h, hint + 100), h, atol=0, rtol=0)

    def test_zero_spatial_map_learns_before_hint_encoder(self):
        torch = self.torch
        layer = self.conditioning.ZeroSpatialResidual(2, 3)
        h = torch.randn(2, 2, 3, 4, requires_grad=True)
        hint = torch.randn(2, 3, 3, 4)
        layer(h, hint).square().mean().backward()
        self.assertGreater(layer.zero_conv.weight.grad.abs().sum().item(), 0)
        for parameter in layer.hint_encoder.parameters():
            self.assertEqual(parameter.grad.abs().sum().item(), 0)
        torch.testing.assert_close(h.grad, 2 * h / h.numel())
        torch.optim.SGD(layer.parameters(), lr=0.1).step()
        layer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(layer(h, hint), h))
        layer(h, hint).square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.hint_encoder.parameters()), 0)

    def test_cfg_interpolates_or_extrapolates_predictions_with_documented_scale(self):
        torch = self.torch
        cfg = self.conditioning.classifier_free_guidance
        uncond = torch.tensor([[1., 2.], [3., 4.]])
        cond = torch.tensor([[3., 1.], [5., 2.]])
        for scale, expected in ((0, uncond), (1, cond),
                                (2, torch.tensor([[5., 0.], [7., 0.]])),
                                (-1, torch.tensor([[-1., 3.], [1., 6.]]))):
            with self.subTest(scale=scale):
                torch.testing.assert_close(cfg(uncond, cond, scale), expected, rtol=0, atol=0)
        torch.testing.assert_close(cfg(uncond, cond), cond, rtol=0, atol=0)

    def test_cfg_remains_differentiable_in_predictions_and_scalar_scale(self):
        torch = self.torch
        uncond = torch.tensor([1., 2.], requires_grad=True)
        cond = torch.tensor([3., 1.], requires_grad=True)
        scale = torch.tensor(2., requires_grad=True)
        self.conditioning.classifier_free_guidance(uncond, cond, scale).sum().backward()
        torch.testing.assert_close(uncond.grad, torch.tensor([-1., -1.]))
        torch.testing.assert_close(cond.grad, torch.tensor([2., 2.]))
        torch.testing.assert_close(scale.grad, torch.tensor(1.))

    def test_invalid_dimensions_and_head_counts_are_rejected(self):
        module = self.conditioning
        for cls in (module.FeatureConcat, module.AddCondition, module.FiLM, module.AdaLN,
                    module.AdaLNZeroBlock, module.CrossAttentionCondition, module.PrefixCondition,
                    module.ZeroSpatialResidual):
            for dimensions in ((0, 2), (2, -1), (True, 2), (2, 1.5)):
                with self.subTest(method=cls.__name__, dimensions=dimensions), self.assertRaises(ValueError):
                    cls(*dimensions)
        for cls in (module.AdaLNZeroBlock, module.CrossAttentionCondition, module.PrefixCondition):
            for heads in (0, True, 1.5, 3):
                with self.subTest(method=cls.__name__, heads=heads), self.assertRaises(ValueError):
                    cls(4, 2, heads=heads)

    def test_global_condition_blocks_reject_malformed_shapes(self):
        torch, module = self.torch, self.conditioning
        h, c = torch.randn(2, 3, 4), torch.randn(2, 2)
        for cls in (module.FeatureConcat, module.AddCondition, module.FiLM, module.AdaLN,
                    module.AdaLNZeroBlock):
            layer = cls(4, 2)
            for data, condition in ((h[:, 0], c), (h[..., :3], c), (h, c[:1]),
                                    (h, c[:, None]), (h[:, :0], c), (h.long(), c),
                                    (h, c.double())):
                with self.subTest(method=cls.__name__, h=data.shape, c=condition.shape), self.assertRaises(ValueError):
                    layer(data, condition)

    def test_token_blocks_reject_malformed_conditions_and_all_masked_rows(self):
        torch, module = self.torch, self.conditioning
        h, c = torch.randn(2, 3, 4), torch.randn(2, 5, 2)
        for cls in (module.CrossAttentionCondition, module.PrefixCondition):
            layer = cls(4, 2)
            for condition in (c[:, 0], c[:1], c[..., :1], c[:, :0], c.long()):
                with self.subTest(method=cls.__name__, shape=condition.shape), self.assertRaises(ValueError):
                    layer(h, condition)
        layer = module.CrossAttentionCondition(4, 2)
        for mask in (torch.zeros(2, 5), torch.zeros(2, 4, dtype=torch.bool),
                     torch.tensor([[False] * 5, [True] * 5])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                layer(h, c, padding_mask=mask)

    def test_spatial_residual_rejects_mismatched_batch_channels_or_resolution(self):
        torch = self.torch
        layer = self.conditioning.ZeroSpatialResidual(2, 3)
        h, hint = torch.randn(2, 2, 3, 4), torch.randn(2, 3, 3, 4)
        for data, condition in ((h[:, 0], hint), (h[:, :1], hint), (h, hint[:1]),
                                (h, hint[:, :2]), (h, hint[..., :3]), (h, hint.long())):
            with self.subTest(h=data.shape, hint=condition.shape), self.assertRaises(ValueError):
                layer(data, condition)

    def test_cfg_rejects_mismatched_predictions_and_nonfinite_or_nonscalar_scale(self):
        torch = self.torch
        cfg = self.conditioning.classifier_free_guidance
        uncond, cond = torch.ones(2, 3), torch.zeros(2, 3)
        for left, right in ((uncond, cond[:1]), (uncond.long(), cond), (uncond, cond.double())):
            with self.subTest(left=left.shape, right=right.shape), self.assertRaises(ValueError):
                cfg(left, right)
        for scale in (float("nan"), float("inf"), "2", torch.ones(2), torch.tensor(float("nan"))):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                cfg(uncond, cond, scale)


if __name__ == "__main__":
    unittest.main()
