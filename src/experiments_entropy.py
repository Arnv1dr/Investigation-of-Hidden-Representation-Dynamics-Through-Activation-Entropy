import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn

import config
from src.activation_metrics import compute_activation_entropy_for_loader, load_bin_edges
from src.data import get_dataloaders
from src.evaluate import evaluate
from src.models import create_model, get_monitored_layer_names
from src.train import train_one_epoch
from src.utils import get_device, set_seed


def run_entropy_experiment(
    model_name: str,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    split_seed: int,
    val_per_class: int,
    bin_edges_dir: str,
    binning_method: str,
    entropy_max_batches: int,
    final_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Runs a training experiment with validation metrics and layerwise activation entropy.

    Entropy is computed after each epoch using fixed bin edges created by pilot_bin_edges.py.
    """
    set_seed(seed)
    device = get_device()

    train_loader, val_loader, test_loader = get_dataloaders(
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

    bin_edges_by_layer = load_bin_edges(
        input_dir=bin_edges_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        layer_names=layer_names,
        method=binning_method,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rows = []

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {epochs}")
    print(f"Training seed: {seed}")
    print(f"Fixed split seed: {split_seed}")
    print(f"Monitored layers: {layer_names}")
    print(f"Entropy max validation batches: {entropy_max_batches}")
    print(f"Using fixed bin edges from: {bin_edges_dir}")

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

        entropy_by_layer = compute_activation_entropy_for_loader(
            model=model,
            data_loader=val_loader,
            layer_names=layer_names,
            bin_edges_by_layer=bin_edges_by_layer,
            device=device,
            max_batches=entropy_max_batches,
        )

        val_generalization_gap = train_accuracy - val_accuracy

        for layer_name in layer_names:
            rows.append({
                "epoch": epoch,
                "model": model_name,
                "dataset": dataset_name,
                "seed": seed,
                "split_seed": split_seed,
                "layer": layer_name,
                "activation_entropy": entropy_by_layer[layer_name],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "val_generalization_gap": val_generalization_gap,
            })

        entropy_summary = " | ".join(
            [f"{layer}: {entropy_by_layer[layer]:.4f}" for layer in layer_names]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Gap: {val_generalization_gap:.4f} | "
            f"Entropy: {entropy_summary}"
        )

    history_df = pd.DataFrame(rows)
    final_test_df = None

    if final_test:
        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        final_train_accuracy = history_df[history_df["epoch"] == epochs]["train_accuracy"].iloc[0]
        test_generalization_gap = final_train_accuracy - test_accuracy

        final_test_df = pd.DataFrame([
            {
                "model": model_name,
                "dataset": dataset_name,
                "seed": seed,
                "split_seed": split_seed,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_generalization_gap": test_generalization_gap,
            }
        ])

        print(
            f"Final Test | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f} | "
            f"Test Gap: {test_generalization_gap:.4f}"
        )

    return history_df, final_test_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn"])
    parser.add_argument("--dataset", type=str, default="FashionMNIST", choices=["MNIST", "FashionMNIST"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--split-seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--val-per-class", type=int, default=config.VAL_PER_CLASS)
    parser.add_argument("--bin-edges-dir", type=str, default="./results/bin_edges")
    parser.add_argument("--binning-method", type=str, default="fd", choices=["fd", "scott", "fixed"])
    parser.add_argument("--entropy-max-batches", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./results/logs")
    parser.add_argument("--final-test", action="store_true")

    args = parser.parse_args()

    history_df, final_test_df = run_entropy_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        split_seed=args.split_seed,
        val_per_class=args.val_per_class,
        bin_edges_dir=args.bin_edges_dir,
        binning_method=args.binning_method,
        entropy_max_batches=args.entropy_max_batches,
        final_test=args.final_test,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / f"{args.model}_{args.dataset}_seed{args.seed}_entropy_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved entropy history to: {history_path}")

    if final_test_df is not None:
        tables_dir = Path(config.TABLES_DIR)
        tables_dir.mkdir(parents=True, exist_ok=True)

        final_test_path = tables_dir / f"{args.model}_{args.dataset}_seed{args.seed}_finaltest.csv"
        final_test_df.to_csv(final_test_path, index=False)
        print(f"Saved final test result to: {final_test_path}")


if __name__ == "__main__":
    main()
