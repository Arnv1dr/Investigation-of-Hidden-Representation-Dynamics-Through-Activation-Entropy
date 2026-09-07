from typing import Literal

import numpy as np
import torch


BinningMethod = Literal["fd", "scott", "fixed"]


def tensor_to_1d_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts an activation tensor into a flattened one-dimensional NumPy array.

    The tensor is detached from the computation graph, moved to CPU, and flattened.
    """
    return tensor.detach().cpu().numpy().ravel()


def compute_bin_edges(
    values: np.ndarray,
    method: BinningMethod = "fd",
    fixed_bins: int = 50,
    padding_ratio: float = 0.01,
) -> np.ndarray:
    """
    Computes histogram bin edges for continuous activation values.

    Parameters:
    - values: flattened activation values
    - method:
        "fd"    = Freedman-Diaconis rule
        "scott" = Scott rule
        "fixed" = fixed number of bins
    - fixed_bins: number of bins used when method="fixed"
    - padding_ratio: expands the outer bin edges slightly to reduce edge clipping

    Returns:
    - NumPy array of bin edges
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Cannot compute bin edges from an empty or non-finite array.")

    if np.all(values == values[0]):
        center = values[0]
        epsilon = 1e-6 if center == 0 else abs(center) * 1e-6
        return np.array([center - epsilon, center + epsilon], dtype=np.float64)

    if method == "fixed":
        bin_edges = np.histogram_bin_edges(values, bins=fixed_bins)
    elif method in ["fd", "scott"]:
        bin_edges = np.histogram_bin_edges(values, bins=method)
    else:
        raise ValueError(f"Unsupported binning method: {method}. Use 'fd', 'scott', or 'fixed'.")

    data_range = bin_edges[-1] - bin_edges[0]
    padding = data_range * padding_ratio if data_range > 0 else 1e-6
    bin_edges[0] -= padding
    bin_edges[-1] += padding

    return bin_edges


def histogram_counts_with_clipping(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Computes histogram counts while clipping values outside the fixed bin range
    into the nearest edge bin.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Cannot create histogram from an empty or non-finite array.")

    left = bin_edges[0]
    right = bin_edges[-1]
    epsilon = np.finfo(np.float64).eps

    clipped_values = np.clip(values, left + epsilon, right - epsilon)
    counts, _ = np.histogram(clipped_values, bins=bin_edges)

    return counts


def compute_shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """
    Computes Shannon entropy from histogram counts.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()

    if total == 0:
        raise ValueError("Cannot compute entropy because histogram count sum is zero.")

    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def compute_shannon_entropy_from_values(
    values: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """
    Computes histogram-based Shannon entropy from activation values.
    """
    counts = histogram_counts_with_clipping(values, bin_edges)
    return compute_shannon_entropy_from_counts(counts)


def compute_activation_entropy(
    activation_tensor: torch.Tensor,
    bin_edges: np.ndarray | None = None,
    method: BinningMethod = "fd",
    fixed_bins: int = 50,
) -> tuple[float, np.ndarray]:
    """
    Computes activation entropy for a single layer activation tensor.

    Returns:
    - entropy value
    - bin edges used
    """
    values = tensor_to_1d_numpy(activation_tensor)

    if bin_edges is None:
        bin_edges = compute_bin_edges(values, method=method, fixed_bins=fixed_bins)

    entropy = compute_shannon_entropy_from_values(values, bin_edges)

    return entropy, bin_edges
