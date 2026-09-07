import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch

import config
from src.models import create_model, get_monitored_layer_names


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_seed_list(seed_string: str) -> list[int]:
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def load_combined_history(aggregated_dir: str) -> pd.DataFrame:
    path = Path(aggregated_dir) / "combined_entropy_history.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.aggregate_results first."
        )
    return pd.read_csv(path)


def load_final_test_summary(aggregated_dir: str) -> pd.DataFrame:
    path = Path(aggregated_dir) / "final_test_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.aggregate_results first."
        )
    return pd.read_csv(path)


def filter_official_main_runs(
    entropy_df: pd.DataFrame,
    official_seeds: list[int],
    expected_epochs: int,
) -> pd.DataFrame:
    """
    Keeps only the official main experiment runs.

    Every model/seed/layer group must contain exactly one observation
    for every expected epoch from 1 through expected_epochs.
    """
    filtered = entropy_df[
        entropy_df["seed"].isin(official_seeds)
    ].copy()

    expected_epoch_set = set(range(1, expected_epochs + 1))

    problems = []

    for (model, dataset, seed, layer), group in filtered.groupby(
        ["model", "dataset", "seed", "layer"]
    ):
        epochs = group["epoch"].tolist()
        actual_epoch_set = set(epochs)

        missing_epochs = sorted(expected_epoch_set - actual_epoch_set)
        unexpected_epochs = sorted(actual_epoch_set - expected_epoch_set)
        duplicate_epochs = sorted(
            group.loc[group["epoch"].duplicated(), "epoch"].unique().tolist()
        )

        if missing_epochs or unexpected_epochs or duplicate_epochs:
            problems.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "layer": layer,
                    "missing_epochs": missing_epochs,
                    "unexpected_epochs": unexpected_epochs,
                    "duplicate_epochs": duplicate_epochs,
                }
            )

    if problems:
        details = "\n".join(str(problem) for problem in problems)
        raise ValueError(
            "Incomplete or invalid official main runs detected:\n"
            + details
        )

    """ Verifies that every model contains all requested official seeds.
    """
    for model in sorted(filtered["model"].unique()):
        actual_seeds = set(
            filtered.loc[filtered["model"] == model, "seed"].unique()
        )
        expected_seeds = set(official_seeds)

        if actual_seeds != expected_seeds:
            missing_seeds = sorted(expected_seeds - actual_seeds)
            unexpected_seeds = sorted(actual_seeds - expected_seeds)

            raise ValueError(
                f"{model.upper()} seed set is incomplete or unexpected. "
                f"Missing seeds: {missing_seeds}; "
                f"Unexpected seeds: {unexpected_seeds}"
            )

    print("Using official seeds only:", official_seeds)
    print("Expected epoch set:", f"1-{expected_epochs}")

    for model in sorted(filtered["model"].unique()):
        model_seeds = sorted(
            filtered.loc[filtered["model"] == model, "seed"].unique()
        )
        print(f"{model.upper()} seeds included:", model_seeds)

    return filtered


def get_performance_history(entropy_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "dataset",
        "seed",
        "split_seed",
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "val_generalization_gap",
    ]

    return entropy_df[columns].drop_duplicates()


def summarize_performance_by_epoch(perf_df: pd.DataFrame) -> pd.DataFrame:
    return (
        perf_df
        .groupby(["model", "dataset", "epoch"], as_index=False)
        .agg(
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


def create_final_validation_summary(perf_df: pd.DataFrame, expected_epochs: int) -> pd.DataFrame:
    final_df = perf_df[perf_df["epoch"] == expected_epochs]

    return (
        final_df
        .groupby(["model", "dataset"], as_index=False)
        .agg(
            final_train_loss_mean=("train_loss", "mean"),
            final_train_loss_std=("train_loss", "std"),
            final_val_loss_mean=("val_loss", "mean"),
            final_val_loss_std=("val_loss", "std"),
            final_train_accuracy_mean=("train_accuracy", "mean"),
            final_train_accuracy_std=("train_accuracy", "std"),
            final_val_accuracy_mean=("val_accuracy", "mean"),
            final_val_accuracy_std=("val_accuracy", "std"),
            final_val_generalization_gap_mean=("val_generalization_gap", "mean"),
            final_val_generalization_gap_std=("val_generalization_gap", "std"),
            n_seeds=("seed", "nunique"),
        )
    )


def create_entropy_summary_with_peak(
    entropy_df: pd.DataFrame,
    expected_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for (model, dataset, seed, layer), group in entropy_df.groupby(
        ["model", "dataset", "seed", "layer"]
    ):
        group = group.sort_values("epoch")

        initial_rows = group[group["epoch"] == 1]
        final_rows = group[group["epoch"] == expected_epochs]

        if initial_rows.empty or final_rows.empty:
            print(
                f"Skipping incomplete group: model={model}, seed={seed}, layer={layer}, "
                f"epochs available={sorted(group['epoch'].unique().tolist())}"
            )
            continue

        initial_entropy = float(initial_rows["activation_entropy"].iloc[0])
        final_entropy = float(final_rows["activation_entropy"].iloc[0])
        peak_entropy = float(group["activation_entropy"].max())
        minimum_entropy = float(group["activation_entropy"].min())
        entropy_change = final_entropy - initial_entropy
        peak_epoch = int(group.loc[group["activation_entropy"].idxmax(), "epoch"])
        minimum_epoch = int(group.loc[group["activation_entropy"].idxmin(), "epoch"])

        rows.append({
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "layer": layer,
            "initial_entropy": initial_entropy,
            "final_entropy": final_entropy,
            "entropy_change": entropy_change,
            "peak_entropy": peak_entropy,
            "peak_epoch": peak_epoch,
            "minimum_entropy": minimum_entropy,
            "minimum_epoch": minimum_epoch,
        })

    per_seed = pd.DataFrame(rows)

    if per_seed.empty:
        raise ValueError("No complete entropy groups found. Check input CSV files and seed filtering.")

    summary = (
        per_seed
        .groupby(["model", "dataset", "layer"], as_index=False)
        .agg(
            initial_entropy_mean=("initial_entropy", "mean"),
            initial_entropy_std=("initial_entropy", "std"),
            final_entropy_mean=("final_entropy", "mean"),
            final_entropy_std=("final_entropy", "std"),
            entropy_change_mean=("entropy_change", "mean"),
            entropy_change_std=("entropy_change", "std"),
            peak_entropy_mean=("peak_entropy", "mean"),
            peak_entropy_std=("peak_entropy", "std"),
            peak_epoch_mean=("peak_epoch", "mean"),
            minimum_entropy_mean=("minimum_entropy", "mean"),
            minimum_entropy_std=("minimum_entropy", "std"),
            minimum_epoch_mean=("minimum_epoch", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    return per_seed, summary


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def create_model_architecture_table() -> pd.DataFrame:
    rows = []

    for model_name in ["mlp", "cnn"]:
        model = create_model(model_name)
        monitored_layers = set(get_monitored_layer_names(model_name))

        if model_name == "mlp":
            layer_specs = [
                ("flatten", "Flatten", "(128, 784)", False),
                ("fc1", "Linear", "(128, 256)", False),
                ("relu1", "ReLU", "(128, 256)", True),
                ("fc2", "Linear", "(128, 128)", False),
                ("relu2", "ReLU", "(128, 128)", True),
                ("fc3", "Linear output layer", "(128, 10)", False),
            ]
        else:
            layer_specs = [
                ("conv1", "Conv2d", "(128, 16, 28, 28)", False),
                ("relu1", "ReLU", "(128, 16, 28, 28)", True),
                ("pool1", "MaxPool2d", "(128, 16, 14, 14)", False),
                ("conv2", "Conv2d", "(128, 32, 14, 14)", False),
                ("relu2", "ReLU", "(128, 32, 14, 14)", True),
                ("pool2", "MaxPool2d", "(128, 32, 7, 7)", False),
                ("flatten", "Flatten", "(128, 1568)", False),
                ("fc1", "Linear", "(128, 128)", False),
                ("relu3", "ReLU", "(128, 128)", True),
                ("fc2", "Linear output layer", "(128, 10)", False),
            ]

        modules = dict(model.named_modules())
        total_params = parameter_count(model)

        for layer_name, layer_type, output_shape, monitored in layer_specs:
            module = modules.get(layer_name)
            params = parameter_count(module) if module is not None else 0

            rows.append({
                "model": model_name,
                "layer": layer_name,
                "layer_type": layer_type,
                "output_shape_batch_128": output_shape,
                "monitored_for_entropy": monitored and layer_name in monitored_layers,
                "trainable_parameters_in_layer": params,
                "total_model_parameters": total_params,
            })

    return pd.DataFrame(rows)


def create_experimental_settings_table(
    entropy_df: pd.DataFrame,
    official_seeds: list[int],
    expected_epochs: int,
) -> pd.DataFrame:
    models = sorted(entropy_df["model"].unique().tolist())

    rows = [
        ("Primary dataset", "Fashion-MNIST"),
        ("Architectures", ", ".join(model.upper() for model in models)),
        ("Main seeds", ", ".join(str(seed) for seed in official_seeds)),
        ("Number of runs per architecture", str(len(official_seeds))),
        ("Epochs per main run", str(expected_epochs)),
        ("Batch size", str(config.BATCH_SIZE)),
        ("Optimizer", "Adam"),
        ("Learning rate", str(config.LEARNING_RATE)),
        ("Loss function", "Cross-entropy loss"),
        ("Activation function", "ReLU"),
        ("Training split", "50,000 images from official training portion"),
        ("Validation split", "10,000 images from official training portion, 1,000 per class"),
        ("Official test set", "10,000 images, evaluated once after each completed run"),
        ("Split seed", str(config.SPLIT_SEED)),
        ("Entropy estimator", "Histogram-based Shannon entropy"),
        ("Main binning method", "Freedman-Diaconis bin edges from training-based pilot"),
        ("Entropy monitoring batches", "79 validation batches"),
        ("Entropy monitoring set", "Full validation split"),
        ("Final output layer included in entropy analysis", "No"),
    ]

    return pd.DataFrame(rows, columns=["setting", "value"])


def plot_train_val_accuracy(perf_summary: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure()

    for model in sorted(perf_summary["model"].unique()):
        df = perf_summary[perf_summary["model"] == model].sort_values("epoch")
        x = df["epoch"]

        plt.plot(x, df["train_accuracy_mean"], label=f"{model.upper()} train")
        plt.fill_between(
            x,
            df["train_accuracy_mean"] - df["train_accuracy_std"].fillna(0),
            df["train_accuracy_mean"] + df["train_accuracy_std"].fillna(0),
            alpha=0.2,
        )

        plt.plot(x, df["val_accuracy_mean"], label=f"{model.upper()} validation")
        plt.fill_between(
            x,
            df["val_accuracy_mean"] - df["val_accuracy_std"].fillna(0),
            df["val_accuracy_mean"] + df["val_accuracy_std"].fillna(0),
            alpha=0.2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy over Training")
    plt.legend()
    plt.grid(True)

    path = plot_dir / "training_validation_accuracy_mean_std.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_train_val_loss(perf_summary: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure()

    for model in sorted(perf_summary["model"].unique()):
        df = perf_summary[perf_summary["model"] == model].sort_values("epoch")
        x = df["epoch"]

        plt.plot(x, df["train_loss_mean"], label=f"{model.upper()} train")
        plt.fill_between(
            x,
            df["train_loss_mean"] - df["train_loss_std"].fillna(0),
            df["train_loss_mean"] + df["train_loss_std"].fillna(0),
            alpha=0.2,
        )

        plt.plot(x, df["val_loss_mean"], label=f"{model.upper()} validation")
        plt.fill_between(
            x,
            df["val_loss_mean"] - df["val_loss_std"].fillna(0),
            df["val_loss_mean"] + df["val_loss_std"].fillna(0),
            alpha=0.2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Training")
    plt.legend()
    plt.grid(True)

    path = plot_dir / "training_validation_loss_mean_std.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_architecture_entropy_comparison(entropy_df: pd.DataFrame, plot_dir: Path) -> None:
    mapping = {
        ("mlp", "relu1"): "MLP early: relu1",
        ("cnn", "relu1"): "CNN early: relu1",
        ("mlp", "relu2"): "MLP late: relu2",
        ("cnn", "relu3"): "CNN late: relu3",
    }

    labeled = entropy_df.copy()
    labeled["comparison_label"] = labeled.apply(
        lambda row: mapping.get((row["model"], row["layer"])), axis=1
    )

    summary = (
        labeled
        .dropna(subset=["comparison_label"])
        .groupby(["comparison_label", "epoch"], as_index=False)
        .agg(
            entropy_mean=("activation_entropy", "mean"),
            entropy_std=("activation_entropy", "std"),
        )
    )

    plt.figure()

    for label in mapping.values():
        df = summary[summary["comparison_label"] == label].sort_values("epoch")
        x = df["epoch"]

        plt.plot(x, df["entropy_mean"], label=label)
        plt.fill_between(
            x,
            df["entropy_mean"] - df["entropy_std"].fillna(0),
            df["entropy_mean"] + df["entropy_std"].fillna(0),
            alpha=0.2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Activation Entropy")
    plt.title("Matched Layer-Position Entropy Comparison")
    plt.legend()
    plt.grid(True)

    path = plot_dir / "architecture_entropy_comparison_mean_std.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_entropy_vs_val_accuracy(entropy_df: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure()

    for (model, layer), df in entropy_df.groupby(["model", "layer"]):
        plt.scatter(
            df["activation_entropy"],
            df["val_accuracy"],
            alpha=0.45,
            label=f"{model.upper()} {layer}",
        )

    plt.xlabel("Activation Entropy")
    plt.ylabel("Validation Accuracy")
    plt.title("Activation Entropy vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    path = plot_dir / "entropy_vs_validation_accuracy.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_entropy_vs_val_gap(entropy_df: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure()

    for (model, layer), df in entropy_df.groupby(["model", "layer"]):
        plt.scatter(
            df["activation_entropy"],
            df["val_generalization_gap"],
            alpha=0.45,
            label=f"{model.upper()} {layer}",
        )

    plt.xlabel("Activation Entropy")
    plt.ylabel("Train - Validation Accuracy")
    plt.title("Activation Entropy vs Validation Generalization Gap")
    plt.legend()
    plt.grid(True)

    path = plot_dir / "entropy_vs_validation_generalization_gap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregated-dir", type=str, default="./results/aggregated")
    parser.add_argument("--output-dir", type=str, default="./results/phase8_tables")
    parser.add_argument("--plot-dir", type=str, default="./results/plots")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--expected-epochs", type=int, default=30)

    args = parser.parse_args()

    official_seeds = parse_seed_list(args.seeds)
    output_dir = ensure_dir(args.output_dir)
    plot_dir = ensure_dir(args.plot_dir)

    raw_entropy_df = load_combined_history(args.aggregated_dir)
    entropy_df = filter_official_main_runs(
        raw_entropy_df,
        official_seeds=official_seeds,
        expected_epochs=args.expected_epochs,
    )
    final_test_summary = load_final_test_summary(args.aggregated_dir)

    perf_df = get_performance_history(entropy_df)
    perf_summary = summarize_performance_by_epoch(perf_df)

    architecture_table = create_model_architecture_table()
    settings_table = create_experimental_settings_table(
        entropy_df,
        official_seeds=official_seeds,
        expected_epochs=args.expected_epochs,
    )
    final_validation_summary = create_final_validation_summary(
        perf_df,
        expected_epochs=args.expected_epochs,
    )
    entropy_per_seed, entropy_summary_with_peak = create_entropy_summary_with_peak(
        entropy_df,
        expected_epochs=args.expected_epochs,
    )

    architecture_table.to_csv(output_dir / "model_architecture_table.csv", index=False)
    settings_table.to_csv(output_dir / "experimental_settings_table.csv", index=False)
    final_validation_summary.to_csv(output_dir / "final_validation_summary.csv", index=False)
    final_test_summary.to_csv(output_dir / "final_test_summary_for_thesis.csv", index=False)
    entropy_per_seed.to_csv(output_dir / "entropy_summary_per_seed.csv", index=False)
    entropy_summary_with_peak.to_csv(output_dir / "entropy_summary_with_peak.csv", index=False)
    perf_summary.to_csv(output_dir / "performance_epoch_summary_mean_std.csv", index=False)

    print(f"Saved table: {output_dir / 'model_architecture_table.csv'}")
    print(f"Saved table: {output_dir / 'experimental_settings_table.csv'}")
    print(f"Saved table: {output_dir / 'final_validation_summary.csv'}")
    print(f"Saved table: {output_dir / 'final_test_summary_for_thesis.csv'}")
    print(f"Saved table: {output_dir / 'entropy_summary_per_seed.csv'}")
    print(f"Saved table: {output_dir / 'entropy_summary_with_peak.csv'}")
    print(f"Saved table: {output_dir / 'performance_epoch_summary_mean_std.csv'}")

    plot_train_val_accuracy(perf_summary, plot_dir)
    plot_train_val_loss(perf_summary, plot_dir)
    plot_architecture_entropy_comparison(entropy_df, plot_dir)
    plot_entropy_vs_val_accuracy(entropy_df, plot_dir)
    plot_entropy_vs_val_gap(entropy_df, plot_dir)

    print()
    print("Final validation summary:")
    print(final_validation_summary.to_string(index=False))

    print()
    print("Entropy summary with peak:")
    print(entropy_summary_with_peak.to_string(index=False))


if __name__ == "__main__":
    main()
