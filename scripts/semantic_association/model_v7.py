import torch
import torch.nn as nn
import math


'''
MLP, but includes the ray rotation matrix, positional embedding and geometric context
'''


class ModelV7(nn.Module):
    def __init__(self, geometric_context_length: int, num_unique_labels: int, geom_feature_length: int):
        super(ModelV7, self).__init__()
        self.geometric_context_length = geometric_context_length
        self.geometric_feature_length = geom_feature_length

        input_length = num_unique_labels + geometric_context_length*geom_feature_length + 9
        self.layer1 = nn.Linear(input_length, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, self.geometric_context_length)

        pe_dim = 2
        position = torch.arange(self.geometric_context_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2) * (-math.log(10000.0) / pe_dim))
        raw_positional_embedding = torch.zeros(self.geometric_context_length, pe_dim)
        raw_positional_embedding[:, 0::2] = torch.sin(position * div_term)
        raw_positional_embedding[:, 1::2] = torch.cos(position * div_term)
        self.positional_embedding = torch.repeat_interleave(raw_positional_embedding[:, 0], repeats=geom_feature_length)

    def forward(self, semantic_label, geometric_features, ray_direction, context_map):
        batch_size = semantic_label.shape[0]
        device = next(self.parameters()).device

        geometric_context = torch.zeros((batch_size, self.geometric_context_length, self.geometric_feature_length), device=device, dtype=torch.float16)
        mask = context_map == -1
        geometric_context[~mask] = geometric_features[context_map[~mask]]
        geometric_context = geometric_context.reshape(batch_size, self.geometric_context_length*self.geometric_feature_length)
        geometric_context += self.positional_embedding.expand(batch_size, self.geometric_context_length*self.geometric_feature_length).to(device)
        input_tensor = torch.cat((semantic_label, geometric_context, ray_direction), dim=1)

        x = torch.sigmoid(self.layer1(input_tensor))
        x = torch.sigmoid(self.layer2(x))
        output = self.layer3(x)
        return output
