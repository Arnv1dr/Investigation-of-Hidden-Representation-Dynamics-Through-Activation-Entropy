from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.entropy import (
    compute_bin_edges,
    compute_shannon_entropy_from_counts,
    histogram_counts_with_clipping,
    tensor_to_1d_numpy,
)
from src.hooks import ActivationRecorder


def sample_values(values: np.ndarray, max_values: int, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly sample values without replacement if the array is larger than max_values.
    """
    if values.size <= max_values:
        return values

    indices = rng.choice(values.size, size=max_values, replace=False)
    return values[indices]


def collect_activation_samples(
    model: nn.Module,
    data_loader: DataLoader,
    layer_names: Iterable[str],
    device: torch.device,
    max_batches: int,
    max_values_per_layer: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Collect sampled activation values from selected layers.

    Used for pilot bin-edge estimation. Keeps memory manageable,
    only a capped number of scalar values per layer is retained.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    layer_samples = defaultdict(list)
    recorder = ActivationRecorder(model=model, layer_names=layer_names)
    recorder.register()

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(tqdm(data_loader, desc="Collecting activation samples", leave=False)):
            if batch_index >= max_batches:
                break

            images = images.to(device)
            recorder.clear()
            _ = model(images)

            activations = recorder.get_activations()

            for layer_name in layer_names:
                values = tensor_to_1d_numpy(activations[layer_name])
                values = sample_values(values, max_values=max_values_per_layer, rng=rng)
                layer_samples[layer_name].append(values)

    recorder.remove()

    combined = {}
    for layer_name in layer_names:
        if not layer_samples[layer_name]:
            raise ValueError(f"No activation samples collected for layer {layer_name}.")
        combined[layer_name] = np.concatenate(layer_samples[layer_name])

    return combined


def create_bin_edges_from_samples(
    layer_samples: Dict[str, np.ndarray],
    method: str = "fd",
    fixed_bins: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Create bin edges per layer from sampled pilot activations.
    """
    bin_edges_by_layer = {}

    for layer_name, values in layer_samples.items():
        bin_edges_by_layer[layer_name] = compute_bin_edges(
            values=values,
            method=method,
            fixed_bins=fixed_bins,
        )

    return bin_edges_by_layer


def save_bin_edges(
    bin_edges_by_layer: Dict[str, np.ndarray],
    output_dir: str,
    model_name: str,
    dataset_name: str,
    method: str,
) -> None:
    """
    Save bin edges as .npy files, one per monitored layer.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for layer_name, bin_edges in bin_edges_by_layer.items():
        file_path = output_path / f"{model_name}_{dataset_name}_{layer_name}_{method}_bin_edges.npy"
        np.save(file_path, bin_edges)


def load_bin_edges(
    input_dir: str,
    model_name: str,
    dataset_name: str,
    layer_names: Iterable[str],
    method: str,
) -> Dict[str, np.ndarray]:
    """
    Load saved bin edges for all monitored layers.
    """
    input_path = Path(input_dir)
    bin_edges_by_layer = {}

    for layer_name in layer_names:
        file_path = input_path / f"{model_name}_{dataset_name}_{layer_name}_{method}_bin_edges.npy"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing bin edge file: {file_path}. "
                f"Run src.pilot_bin_edges first."
            )

        bin_edges_by_layer[layer_name] = np.load(file_path)

    return bin_edges_by_layer


def compute_activation_entropy_for_loader(
    model: nn.Module,
    data_loader: DataLoader,
    layer_names: Iterable[str],
    bin_edges_by_layer: Dict[str, np.ndarray],
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    """
    Compute layerwise activation entropy over a fixed monitoring subset.

    The function accumulates histogram counts across batches and then computes
    one entropy value per layer.
    """
    model.eval()

    counts_by_layer = {
        layer_name: np.zeros(len(bin_edges_by_layer[layer_name]) - 1, dtype=np.float64)
        for layer_name in layer_names
    }

    recorder = ActivationRecorder(model=model, layer_names=layer_names)
    recorder.register()

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(tqdm(data_loader, desc="Computing activation entropy", leave=False)):
            if batch_index >= max_batches:
                break

            images = images.to(device)
            recorder.clear()
            _ = model(images)

            activations = recorder.get_activations()

            for layer_name in layer_names:
                values = tensor_to_1d_numpy(activations[layer_name])
                counts = histogram_counts_with_clipping(
                    values=values,
                    bin_edges=bin_edges_by_layer[layer_name],
                )
                counts_by_layer[layer_name] += counts

    recorder.remove()

    entropy_by_layer = {
        layer_name: compute_shannon_entropy_from_counts(counts)
        for layer_name, counts in counts_by_layer.items()
    }

    return entropy_by_layer
