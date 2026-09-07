import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

import config
from src.activation_metrics import (
    collect_activation_samples,
    create_bin_edges_from_samples,
    save_bin_edges,
)
from src.data import get_dataloaders
from src.evaluate import evaluate
from src.models import create_model, get_monitored_layer_names
from src.train import train_one_epoch
from src.utils import get_device, set_seed


def merge_layer_samples(
    accumulated_samples: dict[str, list[np.ndarray]],
    new_samples: dict[str, np.ndarray],
) -> None:
    """Appends newly collected activation samples to the accumulated per-layer sample."""
    for layer_name, values in new_samples.items():
        accumulated_samples[layer_name].append(values)


def concatenate_layer_samples(
    accumulated_samples: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenates all collected activation samples per layer."""
    combined_samples = {}

    for layer_name, sample_list in accumulated_samples.items():
        if not sample_list:
            raise ValueError(f"No samples collected for layer {layer_name}.")
        combined_samples[layer_name] = np.concatenate(sample_list)

    return combined_samples


def summarize_bin_edges(
    bin_edges_by_layer: dict[str, np.ndarray],
    combined_samples: dict[str, np.ndarray],
) -> list[dict]:
    """Creates a summary table for saved metadata."""
    rows = []

    for layer_name, bin_edges in bin_edges_by_layer.items():
        values = combined_samples[layer_name]

        rows.append({
            "layer": layer_name,
            "sample_count": int(values.size),
            "sample_min": float(np.min(values)),
            "sample_max": float(np.max(values)),
            "num_bins": int(len(bin_edges) - 1),
            "bin_min": float(bin_edges[0]),
            "bin_max": float(bin_edges[-1]),
        })

    return rows


def run_training_based_pilot(
    model_name: str,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    split_seed: int,
    val_per_class: int,
    max_batches: int,
    max_values_per_layer_per_batch: int,
    method: str,
    fixed_bins: int,
    output_dir: str,
) -> None:
    """
    Train a pilot model and create fixed bin edges from pilot activations.

    Procedure:
    1. Initialize a model with the pilot seed.
    2. Train for the requested number of pilot epochs.
    3. After each epoch, collect validation activations from monitored layers.
    4. Aggregate activation samples across pilot epochs.
    5. Compute one fixed histogram bin-edge array per layer.
    6. Save bin edges for reuse in the main entropy experiments.
    """
    set_seed(seed)
    device = get_device()

    train_loader, val_loader, _ = get_dataloaders(
        dataset_name=dataset_name,
        data_dir=config.DATA_DIR,
        batch_size=batch_size,
        split_dir=config.SPLIT_DIR,
        val_per_class=val_per_class,
        split_seed=split_seed,
        train_seed=seed,
        num_workers=config.NUM_WORKERS,
    )

    model = create_model(model_name).to(device)
    layer_names = get_monitored_layer_names(model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accumulated_samples = defaultdict(list)
    pilot_history = []

    print("Creating training-based pilot bin edges")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Pilot epochs: {epochs}")
    print(f"Pilot seed: {seed}")
    print(f"Split seed: {split_seed}")
    print(f"Layers: {layer_names}")
    print(f"Binning method: {method}")
    print(f"Max validation batches per pilot epoch: {max_batches}")
    print(f"Max sampled values per layer per batch: {max_values_per_layer_per_batch}")
    print("Note: this pilot run is not part of the final reported results.")
    print()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Pilot Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        epoch_samples = collect_activation_samples(
            model=model,
            data_loader=val_loader,
            layer_names=layer_names,
            device=device,
            max_batches=max_batches,
            max_values_per_layer=max_values_per_layer_per_batch,
            seed=seed + epoch,
        )

        merge_layer_samples(accumulated_samples, epoch_samples)

        pilot_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

    combined_samples = concatenate_layer_samples(accumulated_samples)

    bin_edges_by_layer = create_bin_edges_from_samples(
        layer_samples=combined_samples,
        method=method,
        fixed_bins=fixed_bins,
    )

    save_bin_edges(
        bin_edges_by_layer=bin_edges_by_layer,
        output_dir=output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        method=method,
    )

    summary_rows = summarize_bin_edges(
        bin_edges_by_layer=bin_edges_by_layer,
        combined_samples=combined_samples,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "purpose": "training_based_pilot_bin_edges",
        "model": model_name,
        "dataset": dataset_name,
        "pilot_epochs": epochs,
        "pilot_seed": seed,
        "split_seed": split_seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_per_class": val_per_class,
        "max_validation_batches_per_epoch": max_batches,
        "max_values_per_layer_per_batch": max_values_per_layer_per_batch,
        "binning_method": method,
        "fixed_bins_if_used": fixed_bins,
        "monitored_layers": layer_names,
        "pilot_history": pilot_history,
        "bin_edge_summary": summary_rows,
        "important_note": (
            "This pilot run is used only to determine fixed histogram bin edges. "
            "It is not included in the final reported thesis results."
        ),
    }

    metadata_path = output_path / f"{model_name}_{dataset_name}_{method}_training_pilot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print("Saved training-based pilot bin edges:")
    for row in summary_rows:
        print(
            f"- {row['layer']}: "
            f"{row['num_bins']} bins | "
            f"samples={row['sample_count']} | "
            f"bin range=({row['bin_min']:.6f}, {row['bin_max']:.6f})"
        )

    print(f"Saved metadata to: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn"])
    parser.add_argument("--dataset", type=str, default="FashionMNIST", choices=["MNIST", "FashionMNIST"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--val-per-class", type=int, default=config.VAL_PER_CLASS)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-values-per-layer-per-batch", type=int, default=10000)
    parser.add_argument("--method", type=str, default="fd", choices=["fd", "scott", "fixed"])
    parser.add_argument("--fixed-bins", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./results/bin_edges")

    args = parser.parse_args()

    run_training_based_pilot(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        split_seed=args.split_seed,
        val_per_class=args.val_per_class,
        max_batches=args.max_batches,
        max_values_per_layer_per_batch=args.max_values_per_layer_per_batch,
        method=args.method,
        fixed_bins=args.fixed_bins,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
