import torch.nn as nn


'''
The simplest geometric feature extractor - just an MLP
'''


class GeomContMlpFeatures(nn.Module):
    def __init__(self, geometric_context_size, context_feature_length):
        super(GeomContMlpFeatures, self).__init__()
        self.activation = nn.ReLU()
        self.input_length = geometric_context_size**3
        self.layer1 = nn.Linear(self.input_length, 256)
        self.layer2 = nn.Linear(256, context_feature_length)

    def forward(self, densities):
        batch_size = densities.shape[0]
        inputs = densities.reshape(batch_size, self.input_length)

        x = self.activation(self.layer1(inputs))
        output = self.layer2(x)
        return output
