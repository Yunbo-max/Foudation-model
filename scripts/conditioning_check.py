"""Structural conditioning checks on explicit fixtures, not generation benchmarks."""
import json

import torch

from fm_tutorial.diffusion.conditioning import (
    FeatureConcat, AddCondition, FiLM, AdaLN, AdaLNZeroBlock,
    CrossAttentionCondition, PrefixCondition, ZeroSpatialResidual,
    classifier_free_guidance,
)


def main():
    torch.manual_seed(23)
    torch.set_num_threads(1)
    h = torch.randn(2, 4, 8)
    c = torch.randn(2, 3)
    context = torch.randn(2, 5, 3)
    report = {"scope": "small synthetic mathematical fixtures, no model-quality result", "output_shapes": {}}
    for cls in (FeatureConcat, AddCondition, FiLM, AdaLN):
        report["output_shapes"][cls.__name__] = list(cls(8, 3)(h, c).shape)

    concat, addition = FeatureConcat(8, 3), AddCondition(8, 3)
    # Special case Wh=I: concatenate+Linear can exactly match h+Wc*c+b.
    # General Wh needs a learned data projection as well; addition alone fixes Wh=I.
    with torch.no_grad():
        concat.proj.weight[:, :8].copy_(torch.eye(8))
        concat.proj.weight[:, 8:].copy_(addition.proj.weight)
        concat.proj.bias.copy_(addition.proj.bias)
    report["concat_projected_sum_error"] = float((concat(h, c) - addition(h, c)).abs().max().detach())
    assert report["concat_projected_sum_error"] < 1e-5

    zero = AdaLNZeroBlock(8, 3, heads=2)
    report["adaln_zero_identity_error"] = float((zero(h, c) - h).abs().max().detach())
    assert report["adaln_zero_identity_error"] == 0
    # One real update demonstrates that zero initialization is not permanent.
    optimizer = torch.optim.SGD(zero.parameters(), lr=0.01)
    optimizer.zero_grad()
    zero(h, c).square().mean().backward()
    optimizer.step()
    report["adaln_zero_change_after_one_update"] = float((zero(h, c) - h).abs().max().detach())

    cross = CrossAttentionCondition(8, 3, heads=2).eval()
    output = cross(h, context)
    report["cross_attention_condition_change"] = float((output - cross(h, context + 1)).abs().max().detach())
    report["output_shapes"]["CrossAttentionCondition"] = list(output.shape)
    report["output_shapes"]["PrefixCondition"] = list(PrefixCondition(8, 3, heads=2)(h, context).shape)

    feature, hint = torch.randn(2, 4, 8, 8), torch.randn(2, 1, 8, 8)
    spatial = ZeroSpatialResidual(4, 1)
    report["zero_spatial_identity_error"] = float((spatial(feature, hint) - feature).abs().max().detach())
    assert report["zero_spatial_identity_error"] == 0
    report["output_shapes"]["ZeroSpatialResidual"] = list(feature.shape)

    unconditional, conditional = torch.randn_like(h), torch.randn_like(h)
    report["cfg_scale_zero_matches_unconditional"] = bool(torch.allclose(
        classifier_free_guidance(unconditional, conditional, 0), unconditional))
    report["cfg_scale_one_matches_conditional"] = bool(torch.allclose(
        classifier_free_guidance(unconditional, conditional, 1), conditional))
    assert report["cfg_scale_zero_matches_unconditional"] and report["cfg_scale_one_matches_conditional"]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
