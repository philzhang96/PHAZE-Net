import torch
from models.hierarchical_agfw import HierarchicalAGFW

def test_hierarchical_agfw():
    batch_size = 2
    num_regions = 5
    feature_dim = 128

    # Simulate temporal features at 3 scales
    short_feats = torch.rand(batch_size, num_regions, feature_dim)
    medium_feats = torch.rand(batch_size, num_regions, feature_dim)
    long_feats = torch.rand(batch_size, num_regions, feature_dim)

    model = HierarchicalAGFW(input_dim=feature_dim)
    fused_output, attn_weights = model(short_feats, medium_feats, long_feats)

    print("✅ Fused output shape:", fused_output.shape)  # Expect: (B, F)
    print("🔍 Short weights shape:", attn_weights["short_weights"].shape)  # (B, R, 1)
    print("🔍 Medium weights shape:", attn_weights["medium_weights"].shape)
    print("🔍 Long weights shape:", attn_weights["long_weights"].shape)

    assert fused_output.shape == (batch_size, feature_dim)
    assert attn_weights["short_weights"].shape == (batch_size, num_regions, 1)
    assert attn_weights["medium_weights"].shape == (batch_size, num_regions, 1)
    assert attn_weights["long_weights"].shape == (batch_size, num_regions, 1)
    print("✅ All tests passed.")

if __name__ == "__main__":
    test_hierarchical_agfw()
