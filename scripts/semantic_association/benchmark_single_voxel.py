import torch.nn as nn
import torch


'''
This is one of the benchmark solutions. The semantic label is to be assigned to the first voxel along the ray that has 
density higher then some threshold.
'''


class BenchmarkSingleVoxel(nn.Module):
    def __init__(self, density_threshold: float):
        super(BenchmarkSingleVoxel, self).__init__()
        self.density_threshold = density_threshold

    def forward(self, semantic_label, geometry, ray_direction, output_guess=None):
        mask = geometry >= self.density_threshold
        first_occurrence = mask.cumsum(dim=1).cumsum(dim=1).eq(1).float()
        return first_occurrence