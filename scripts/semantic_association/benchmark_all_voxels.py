import torch.nn as nn


'''
The semantic label is to be assigned to all voxels with density larger than 0
'''


class BenchmarkAllVoxels(nn.Module):
    def __init__(self):
        super(BenchmarkAllVoxels, self).__init__()

    def forward(self, semantic_label, voxel_densities, ray_directions):
        output = voxel_densities > 1e-5
        return output.float()