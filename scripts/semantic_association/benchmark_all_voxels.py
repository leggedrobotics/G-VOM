import torch.nn as nn


'''
This is one of the benchmark solutions. The semantic label is to be assigned to all voxels with density larger than 0
'''


class BenchmarkAllVoxels(nn.Module):
    def __init__(self):
        super(BenchmarkAllVoxels, self).__init__()

    def forward(self, semantic_label, geometry, ray_directions):
        output = geometry > 1e-5
        return output.float()