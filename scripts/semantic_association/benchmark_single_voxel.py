import torch
import torch.nn as nn


'''
The semantic label is to be assigned to the first voxel along the ray that has a density higher then the specified threshold.
'''


class BenchmarkSingleVoxel(nn.Module):
    def __init__(self, density_threshold: float):
        super(BenchmarkSingleVoxel, self).__init__()
        self.density_threshold = density_threshold

    def forward(self, semantic_label, voxel_densities, ray_direction):
        mask = voxel_densities >= self.density_threshold
        first_occurrence = mask.cumsum(dim=1).cumsum(dim=1).eq(1).half()
        return first_occurrence