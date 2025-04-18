import torch
from models.agfw import AGFW

# Dummy input simulating feature output from CNN over ROIs
batch_size = 2
chunk_size = 5
num_regions = 20
feature_dim = 64

dummy_input = torch.rand(batch_size, chunk_size, num_regions, feature_dim)

# Instantiate AGFW
agfw = AGFW(num_regions=num_regions, feature_dim=feature_dim)

# Forward pass
output = agfw(dummy_input)

print("✅ AGFW forward pass complete.")
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")
