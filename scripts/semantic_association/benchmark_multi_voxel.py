import torch
import torch.nn as nn


'''
The semantic label is to be assigned to the first continuous group of voxels that have a density higher than the specified threshold.
'''


class BenchmarkMultiVoxel(nn.Module):
    def __init__(self, density_threshold: float):
        super(BenchmarkMultiVoxel, self).__init__()
        self.density_threshold = density_threshold

    def forward(self, semantic_label, voxel_densities, ray_directions):
        mask = voxel_densities > self.density_threshold
        batch_size = voxel_densities.shape[0]
        context_length = voxel_densities.shape[1]

        output = torch.zeros(voxel_densities.shape, dtype=torch.bool, device=voxel_densities.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=voxel_densities.device)
        seen_one = torch.zeros(batch_size, dtype=torch.bool, device=voxel_densities.device)
        for col in range(context_length):
            output[:, col] = mask[:, col] & active
            seen_one |= mask[:, col]
            active &= (~seen_one) | (seen_one & mask[:, col])
            if (~active).all():
                break
        return output.half()