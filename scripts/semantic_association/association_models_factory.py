from typing import Tuple
import torch

from semantic_association.benchmark_single_voxel import BenchmarkSingleVoxel
from semantic_association.benchmark_multi_voxel import BenchmarkMultiVoxel
from semantic_association.benchmark_all_voxels import BenchmarkAllVoxels
from semantic_association.model_v1 import ModelV1
from semantic_association.model_v6 import ModelV6

from semantic_association.mlp_geom_context import GeomContMlpFeatures



def get_trained_model(model_version: str, model_weights_path: str, feature_version: str=None, feature_weights_path: str="")\
        -> Tuple[torch.nn.Module, torch.nn.Module, float, int]:
    geometric_context_length = 128
    geometric_feature_length = 16
    num_labels = 52

    feature_extractor = None
    skip_pixels = 2

    if model_version == "Single":
        model = BenchmarkSingleVoxel(0.2)
        place_label_threshold = 0.5
    elif model_version == "Multi":
        model = BenchmarkMultiVoxel(1e-5)
        place_label_threshold = 0.5
    elif model_version == "All":
        model = BenchmarkAllVoxels()
        place_label_threshold = 0.5
    elif model_version == "v1":
        model = ModelV1(geometric_context_length, num_labels)
        place_label_threshold = 0.0
    elif model_version == "v6":
        model = ModelV6(geometric_context_length, num_labels, geometric_feature_length)
        feature_extractor = get_feature_extractor(feature_version, feature_weights_path)
        place_label_threshold = 0.0
        skip_pixels = 4
    else:
        print(f"[ERROR] Unknown model version '{model_version}'!")
        exit(1)

    if model_version[0] == 'v':
        model.load_state_dict(torch.load(model_weights_path))
    return  model, feature_extractor, place_label_threshold, skip_pixels

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
