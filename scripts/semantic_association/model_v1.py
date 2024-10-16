import torch
import torch.nn as nn


'''
The simplest version of a 2D to 3D association network. It is just a two layer MLP. The input is the geometric context and the
label concatenated.
'''


class ModelV1(nn.Module):
    def __init__(self, geometric_context_length: int, num_unique_labels: int):
        super(ModelV1, self).__init__()
        self.geometric_context_length = geometric_context_length

        input_length = num_unique_labels + geometric_context_length
        self.layer1 = nn.Linear(input_length, input_length)
        self.layer2 = nn.Linear(input_length, self.geometric_context_length)

    def forward(self, semantic_label, geometry, ray_direction):
        if len(semantic_label.shape) == 1:
            semantic_label = semantic_label[None, :]
        input_tensor = torch.cat((semantic_label, geometry), dim=1)
        x = self.layer1(input_tensor)
        x = torch.sigmoid(x)
        output = self.layer2(x)
        return output
