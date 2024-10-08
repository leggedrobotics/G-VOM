import torch
import torch.nn as nn
from association_2d_3d.feature_extractors.positional_encodings import get_sinusoidal_pe


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

        raw_positional_embedding = get_sinusoidal_pe(self.geometric_context_length)[:, 0]
        self.positional_embedding = torch.repeat_interleave(raw_positional_embedding, repeats=geom_feature_length)

    def forward(self, semantic_label, geometric_features, ray_direction, context_map, output_guess=None):
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

    # Prints out a warning if the gradient is too big or too small
    def check_gradient(self):
        abs_layer1_weight_grad = torch.abs(self.layer1.weight.grad)
        layer1_weight_max = torch.max(abs_layer1_weight_grad)
        abs_layer1_bias_grad = torch.abs(self.layer1.bias.grad)
        layer1_bias_max = torch.max(abs_layer1_bias_grad)
        abs_layer2_weight_grad = torch.abs(self.layer2.weight.grad)
        layer2_weight_max = torch.max(abs_layer2_weight_grad)
        abs_layer2_bias_grad = torch.abs(self.layer2.bias.grad)
        layer2_bias_max = torch.max(abs_layer2_bias_grad)
        abs_layer3_weight_grad = torch.abs(self.layer3.weight.grad)
        layer3_weight_max = torch.max(abs_layer3_weight_grad)
        abs_layer3_bias_grad = torch.abs(self.layer3.bias.grad)
        layer3_bias_max = torch.max(abs_layer3_bias_grad)

        max_grad_layer1 = min(layer1_weight_max, layer1_bias_max)
        max_grad_layer2 = min(layer2_weight_max, layer2_bias_max)
        max_grad_layer3 = min(layer3_weight_max, layer3_bias_max)
        max_grad = min(max_grad_layer1, max_grad_layer2, max_grad_layer3)
        if max_grad < 1e-7:
            print(f"[WARNING] The gradient of model V7 is very small! Maximum is: {max_grad}")
        if max_grad > 1e6:
            print(f"[WARNING] The gradient of model V7 is too large! Maximum is: {max_grad}")
