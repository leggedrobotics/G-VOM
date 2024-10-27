import torch
import torch.nn as nn


'''
This is one of the benchmark solutions. The semantic label is to be assigned to the first voxel along the ray that has 
density higher then some threshold.
'''


class BenchmarkMultiVoxel(nn.Module):
    def __init__(self, density_threshold: float):
        super(BenchmarkMultiVoxel, self).__init__()
        self.density_threshold = density_threshold

    def forward(self, semantic_label, geometry, ray_directions, ground_truth_guess=None):
        mask = geometry > self.density_threshold
        batch_size = geometry.shape[0]
        context_length = geometry.shape[1]

        output = torch.zeros(geometry.shape, dtype=torch.bool, device=geometry.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=geometry.device)
        seen_one = torch.zeros(batch_size, dtype=torch.bool, device=geometry.device)
        for col in range(context_length):
            output[:, col] = mask[:, col] & active
            seen_one |= mask[:, col]
            active &= (~seen_one) | (seen_one & mask[:, col])
            if (~active).all():
                break
        return output.float()