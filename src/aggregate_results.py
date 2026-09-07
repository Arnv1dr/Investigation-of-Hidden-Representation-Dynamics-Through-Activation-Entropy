import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OFFICIAL_SEEDS = {1, 2, 3, 4, 5}
EXPECTED_EPOCHS = 30
OFFICIAL_MODELS = {"mlp", "cnn"}
OFFICIAL_DATASET = "FashionMNIST"

def load_entropy_histories(log_dir: str) -> pd.DataFrame:
    """
    Load official entropy-history CSV files.

    Official thesis runs:
    - models: MLP and CNN
    - dataset: Fashion-MNIST
    - seeds: 1-5
    - 30 complete epochs
    """
    paths = sorted(Path(log_dir).glob("*_entropy_history.csv"))

    if not paths:
        raise FileNotFoundError(
            f"No entropy history CSV files found in {log_dir}"
        )

    frames = []

    for path in paths:
        df = pd.read_csv(path)

        
        df = df[
            df["seed"].isin(OFFICIAL_SEEDS)
            & df["model"].isin(OFFICIAL_MODELS)
            & (df["dataset"] == OFFICIAL_DATASET)
        ].copy()

        if df.empty:
            continue

        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError(
            "No official entropy histories for seeds 1-5 were found."
        )

    combined = pd.concat(frames, ignore_index=True)

    
    run_lengths = (
        combined
        .groupby(["model", "dataset", "seed"])["epoch"]
        .nunique()
    )

    incomplete = run_lengths[run_lengths != EXPECTED_EPOCHS]

    if not incomplete.empty:
        raise ValueError(
            "Incomplete official entropy runs detected:\n"
            + incomplete.to_string()
        )

    return combined


def load_final_test_results(table_dir: str) -> pd.DataFrame:
    """
    Load official final-test results for seeds 1-5.
    """
    paths = sorted(Path(table_dir).glob("*_finaltest.csv"))

    if not paths:
        raise FileNotFoundError(
            f"No final test CSV files found in {table_dir}"
        )

    frames = []

    for path in paths:
        df = pd.read_csv(path)

        df = df[
            df["seed"].isin(OFFICIAL_SEEDS)
            & df["model"].isin(OFFICIAL_MODELS)
            & (df["dataset"] == OFFICIAL_DATASET)
        ].copy()

        if df.empty:
            continue

        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError(
            "No official final-test results for seeds 1-5 were found."
        )

    combined = pd.concat(frames, ignore_index=True)

    
    expected_runs = {
        (model, seed)
        for model in OFFICIAL_MODELS
        for seed in OFFICIAL_SEEDS
    }

    actual_runs = set(
        zip(combined["model"], combined["seed"])
    )

    missing_runs = expected_runs - actual_runs

    if missing_runs:
        raise ValueError(
            f"Missing official final-test runs: {sorted(missing_runs)}"
        )

    return combined


def create_epoch_summary(entropy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate epoch-wise metrics across seeds.

    Because entropy history has one row per epoch/layer, performance metrics are
    repeated once per layer. To avoid duplication for performance summaries,
    this function still groups by model, epoch, and layer because the entropy
    metric is layer-specific.
    """
    grouped = (
        entropy_df
        .groupby(["model", "dataset", "epoch", "layer"], as_index=False)
        .agg(
            activation_entropy_mean=("activation_entropy", "mean"),
            activation_entropy_std=("activation_entropy", "std"),
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            train_accuracy_mean=("train_accuracy", "mean"),
            train_accuracy_std=("train_accuracy", "std"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            val_generalization_gap_mean=("val_generalization_gap", "mean"),
            val_generalization_gap_std=("val_generalization_gap", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    return grouped


def create_final_epoch_summary(entropy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final-epoch summary per model and layer.
    Includes initial entropy, final entropy, entropy change, final validation metrics.
    """
    max_epoch = EXPECTED_EPOCHS

    initial_df = entropy_df[entropy_df["epoch"] == 1]
    final_df = entropy_df[entropy_df["epoch"] == EXPECTED_EPOCHS]

    initial_entropy = (
        initial_df
        .groupby(["model", "dataset", "layer"], as_index=False)
        .agg(initial_entropy_mean=("activation_entropy", "mean"),
             initial_entropy_std=("activation_entropy", "std"))
    )

    final_summary = (
        final_df
        .groupby(["model", "dataset", "layer"], as_index=False)
        .agg(
            final_entropy_mean=("activation_entropy", "mean"),
            final_entropy_std=("activation_entropy", "std"),
            final_train_accuracy_mean=("train_accuracy", "mean"),
            final_train_accuracy_std=("train_accuracy", "std"),
            final_val_accuracy_mean=("val_accuracy", "mean"),
            final_val_accuracy_std=("val_accuracy", "std"),
            final_val_gap_mean=("val_generalization_gap", "mean"),
            final_val_gap_std=("val_generalization_gap", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    summary = final_summary.merge(
        initial_entropy,
        on=["model", "dataset", "layer"],
        how="left",
    )

    summary["entropy_change_mean"] = (
        summary["final_entropy_mean"] - summary["initial_entropy_mean"]
    )

    return summary


def create_final_test_summary(final_test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate final test results across seeds.
    """
    summary = (
        final_test_df
        .groupby(["model", "dataset"], as_index=False)
        .agg(
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_generalization_gap_mean=("test_generalization_gap", "mean"),
            test_generalization_gap_std=("test_generalization_gap", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    return summary


def plot_entropy_curves(epoch_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Plot mean activation entropy over epochs with std bands for each model.
    One plot is created per model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model in sorted(epoch_summary["model"].unique()):
        model_df = epoch_summary[epoch_summary["model"] == model]

        plt.figure()

        for layer in sorted(model_df["layer"].unique()):
            layer_df = model_df[model_df["layer"] == layer].sort_values("epoch")

            x = layer_df["epoch"]
            y = layer_df["activation_entropy_mean"]
            y_std = layer_df["activation_entropy_std"].fillna(0)

            plt.plot(x, y, label=layer)
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel("Activation Entropy")
        plt.title(f"{model.upper()} Activation Entropy over Training")
        plt.legend()
        plt.grid(True)

        file_path = output_path / f"{model}_activation_entropy_mean_std.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {file_path}")


def plot_validation_accuracy(epoch_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Plot mean validation accuracy over epochs for each model.

    Because performance metrics are repeated across layer rows, first
    duplicate model/seed/epoch rows are removed before aggregating.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    perf_df = (
        epoch_summary
        .groupby(["model", "dataset", "epoch"], as_index=False)
        .agg(
            val_accuracy_mean=("val_accuracy_mean", "mean"),
            val_accuracy_std=("val_accuracy_std", "mean"),
            train_accuracy_mean=("train_accuracy_mean", "mean"),
            train_accuracy_std=("train_accuracy_std", "mean"),
            val_gap_mean=("val_generalization_gap_mean", "mean"),
            val_gap_std=("val_generalization_gap_std", "mean"),
        )
    )

    plt.figure()

    for model in sorted(perf_df["model"].unique()):
        model_df = perf_df[perf_df["model"] == model].sort_values("epoch")
        x = model_df["epoch"]
        y = model_df["val_accuracy_mean"]
        y_std = model_df["val_accuracy_std"].fillna(0)

        plt.plot(x, y, label=f"{model.upper()} Validation Accuracy")
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy over Training")
    plt.legend()
    plt.grid(True)

    file_path = output_path / "validation_accuracy_mean_std.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {file_path}")


def plot_generalization_gap(epoch_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Plots mean validation generalization gap over epochs for each model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    perf_df = (
        epoch_summary
        .groupby(["model", "dataset", "epoch"], as_index=False)
        .agg(
            val_gap_mean=("val_generalization_gap_mean", "mean"),
            val_gap_std=("val_generalization_gap_std", "mean"),
        )
    )

    plt.figure()

    for model in sorted(perf_df["model"].unique()):
        model_df = perf_df[perf_df["model"] == model].sort_values("epoch")
        x = model_df["epoch"]
        y = model_df["val_gap_mean"]
        y_std = model_df["val_gap_std"].fillna(0)

        plt.plot(x, y, label=f"{model.upper()} Gap")
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Train - Validation Accuracy")
    plt.title("Validation Generalization Gap over Training")
    plt.legend()
    plt.grid(True)

    file_path = output_path / "validation_generalization_gap_mean_std.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="./results/logs")
    parser.add_argument("--table-dir", type=str, default="./results/tables")
    parser.add_argument("--output-dir", type=str, default="./results/aggregated")
    parser.add_argument("--plot-dir", type=str, default="./results/plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entropy_df = load_entropy_histories(args.log_dir)
    final_test_df = load_final_test_results(args.table_dir)

    epoch_summary = create_epoch_summary(entropy_df)
    final_epoch_summary = create_final_epoch_summary(entropy_df)
    final_test_summary = create_final_test_summary(final_test_df)

    combined_history_path = output_dir / "combined_entropy_history.csv"
    combined_final_test_path = output_dir / "combined_final_test_results.csv"
    epoch_summary_path = output_dir / "epoch_summary_mean_std.csv"
    final_epoch_summary_path = output_dir / "final_epoch_summary.csv"
    final_test_summary_path = output_dir / "final_test_summary.csv"

    entropy_df.to_csv(combined_history_path, index=False)
    final_test_df.to_csv(combined_final_test_path, index=False)
    epoch_summary.to_csv(epoch_summary_path, index=False)
    final_epoch_summary.to_csv(final_epoch_summary_path, index=False)
    final_test_summary.to_csv(final_test_summary_path, index=False)

    print(f"Saved: {combined_history_path}")
    print(f"Saved: {combined_final_test_path}")
    print(f"Saved: {epoch_summary_path}")
    print(f"Saved: {final_epoch_summary_path}")
    print(f"Saved: {final_test_summary_path}")

    plot_entropy_curves(epoch_summary, args.plot_dir)
    plot_validation_accuracy(epoch_summary, args.plot_dir)
    plot_generalization_gap(epoch_summary, args.plot_dir)

    print()
    print("Final test summary:")
    print(final_test_summary.to_string(index=False))

    print()
    print("Final epoch entropy summary:")
    print(final_epoch_summary.to_string(index=False))


if __name__ == "__main__":
    main()
