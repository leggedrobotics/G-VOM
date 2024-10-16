from typing import Tuple
import torch

from semantic_association.benchmark_single_voxel import BenchmarkSingleVoxel
from semantic_association.model_v1 import ModelV1
from semantic_association.model_v7 import ModelV7

from semantic_association.mlp_geom_context import GeomContMlpFeatures



def get_trained_model(model_version: str, num_labels: int, model_weights_path: str, feature_version: str=None,
                      feature_weights_path: str="") -> Tuple[torch.nn.Module, torch.nn.Module, float]:
    geometric_context_length = 128
    geometric_feature_length = 16
    feature_extractor = None

    if model_version == "Single":
        model = BenchmarkSingleVoxel(0.3)
        place_label_threshold = 0.5
    elif model_version == "v1":
        model = ModelV1(geometric_context_length, num_labels)
        place_label_threshold = 0.0
    elif model_version == "v7":
        model = ModelV7(geometric_context_length, num_labels, geometric_feature_length)
        feature_extractor = get_feature_extractor(feature_version, feature_weights_path)
        place_label_threshold = 0.0
    else:
        print(f"[ERROR] Unknown model version '{model_version}'!")
        exit(1)

    if model_version != "Single":
        model.load_state_dict(torch.load(model_weights_path))
    return  model, feature_extractor, place_label_threshold

def get_feature_extractor(extractor_version: str, feature_weights_path: str) -> torch.nn.Module:
    geometric_feature_length = 16
    geometric_context_size = 9

    if extractor_version == "mlp":
        feature_extractor = GeomContMlpFeatures(geometric_context_size, geometric_feature_length)
    else:
        print(f"[ERROR] Unknown feature extractor version '{extractor_version}'")
        exit(1)

    feature_extractor.load_state_dict(torch.load(feature_weights_path))
    return feature_extractor
