import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_seed_list(seed_string: str) -> list[int]:
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def load_performance_summary(phase8_dir: str) -> pd.DataFrame:
    path = Path(phase8_dir) / "performance_epoch_summary_mean_std.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.create_phase8_outputs first."
        )
    return pd.read_csv(path)


def load_entropy_history(aggregated_dir: str) -> pd.DataFrame:
    path = Path(aggregated_dir) / "combined_entropy_history.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.aggregate_results first."
        )
    return pd.read_csv(path)


def load_entropy_summary(phase8_dir: str) -> pd.DataFrame:
    path = Path(phase8_dir) / "entropy_summary_with_peak.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.create_phase8_outputs first."
        )
    return pd.read_csv(path)


def filter_official_main_runs(
    entropy_history: pd.DataFrame,
    official_seeds: list[int],
    expected_epochs: int,
) -> pd.DataFrame:
    """
    Keep only official complete main runs.
    """
    filtered = entropy_history[
        entropy_history["seed"].isin(official_seeds)
        & (entropy_history["epoch"] <= expected_epochs)
    ].copy()

    group_epochs = (
        filtered
        .groupby(["model", "dataset", "seed", "layer"])["epoch"]
        .max()
        .reset_index(name="max_epoch")
    )

    complete_groups = group_epochs[group_epochs["max_epoch"] == expected_epochs][
        ["model", "dataset", "seed", "layer"]
    ]

    filtered = filtered.merge(
        complete_groups,
        on=["model", "dataset", "seed", "layer"],
        how="inner",
    )

    print("Using official seeds only:", official_seeds)
    print("Expected epochs:", expected_epochs)
    for model in sorted(filtered["model"].unique()):
        model_seeds = sorted([int(seed) for seed in filtered[filtered["model"] == model]["seed"].unique()])
        print(f"{model.upper()} seeds included:", model_seeds)

    return filtered


def create_signature_entropy_summary(entropy_history: pd.DataFrame) -> pd.DataFrame:
    mask = (
        ((entropy_history["model"] == "mlp") & (entropy_history["layer"] == "relu1"))
        |
        ((entropy_history["model"] == "cnn") & (entropy_history["layer"] == "relu2"))
    )
    filtered = entropy_history[mask].copy()

    summary = (
        filtered
        .groupby(["model", "layer", "epoch"], as_index=False)
        .agg(
            activation_entropy_mean=("activation_entropy", "mean"),
            activation_entropy_std=("activation_entropy", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    summary["series_label"] = summary.apply(
        lambda row: "MLP relu1 entropy" if row["model"] == "mlp" else "CNN relu2 entropy",
        axis=1,
    )
    return summary


def create_signature_validation_entropy_plot(
    performance_summary: pd.DataFrame,
    signature_entropy_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax1 = plt.subplots()

    for model in ["mlp", "cnn"]:
        df = performance_summary[performance_summary["model"] == model].sort_values("epoch")
        x = df["epoch"]
        label = f"{model.upper()} validation accuracy"

        ax1.plot(x, df["val_accuracy_mean"], label=label)
        ax1.fill_between(
            x,
            df["val_accuracy_mean"] - df["val_accuracy_std"].fillna(0),
            df["val_accuracy_mean"] + df["val_accuracy_std"].fillna(0),
            alpha=0.2,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")

    ax2 = ax1.twinx()

    for model in ["mlp", "cnn"]:
        df = signature_entropy_summary[signature_entropy_summary["model"] == model].sort_values("epoch")
        x = df["epoch"]
        label = "MLP relu1 entropy" if model == "mlp" else "CNN relu2 entropy"

        ax2.plot(x, df["activation_entropy_mean"], linestyle="--", label=label)
        ax2.fill_between(
            x,
            df["activation_entropy_mean"] - df["activation_entropy_std"].fillna(0),
            df["activation_entropy_mean"] + df["activation_entropy_std"].fillna(0),
            alpha=0.15,
        )

    ax2.set_ylabel("Activation Entropy")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Validation Accuracy and Signature Entropy Trajectories")
    ax1.grid(True)

    path = output_dir / "validation_accuracy_vs_signature_entropy.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def create_entropy_change_effect_size_plot(
    entropy_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_df = entropy_summary.copy()
    plot_df["label"] = plot_df["model"].str.upper() + " " + plot_df["layer"]

    order = [
        "MLP relu1",
        "MLP relu2",
        "CNN relu1",
        "CNN relu2",
        "CNN relu3",
    ]
    plot_df["order_key"] = pd.Categorical(plot_df["label"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("order_key")

    x = range(len(plot_df))

    plt.figure()
    plt.bar(
        x,
        plot_df["entropy_change_mean"],
        yerr=plot_df["entropy_change_std"],
        capsize=5,
    )

    plt.xticks(list(x), plot_df["label"], rotation=45, ha="right")
    plt.xlabel("Model and layer")
    plt.ylabel("Mean entropy change (final - initial)")
    plt.title("Entropy-Change Effect Sizes with Error Bars")
    plt.grid(True)

    path = output_dir / "entropy_change_effect_sizes.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def save_companion_tables(
    signature_entropy_summary: pd.DataFrame,
    entropy_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    signature_path = output_dir / "signature_entropy_epoch_summary.csv"
    effect_path = output_dir / "entropy_change_effect_sizes.csv"

    signature_entropy_summary.to_csv(signature_path, index=False)

    effect_df = entropy_summary.copy()
    effect_df["label"] = effect_df["model"].str.upper() + " " + effect_df["layer"]
    effect_df = effect_df[
        [
            "label",
            "model",
            "dataset",
            "layer",
            "initial_entropy_mean",
            "final_entropy_mean",
            "entropy_change_mean",
            "entropy_change_std",
            "peak_entropy_mean",
            "peak_epoch_mean",
            "n_seeds",
        ]
    ]
    effect_df.to_csv(effect_path, index=False)

    print(f"Saved table: {signature_path}")
    print(f"Saved table: {effect_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase8-dir", type=str, default="./results/phase8_tables")
    parser.add_argument("--aggregated-dir", type=str, default="./results/aggregated")
    parser.add_argument("--plot-dir", type=str, default="./results/plots")
    parser.add_argument("--output-dir", type=str, default="./results/summary_figures")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--expected-epochs", type=int, default=30)
    args = parser.parse_args()

    official_seeds = parse_seed_list(args.seeds)

    plot_dir = ensure_dir(args.plot_dir)
    output_dir = ensure_dir(args.output_dir)

    performance_summary = load_performance_summary(args.phase8_dir)
    raw_entropy_history = load_entropy_history(args.aggregated_dir)
    entropy_history = filter_official_main_runs(
        raw_entropy_history,
        official_seeds=official_seeds,
        expected_epochs=args.expected_epochs,
    )
    entropy_summary = load_entropy_summary(args.phase8_dir)

    signature_entropy_summary = create_signature_entropy_summary(entropy_history)

    save_companion_tables(signature_entropy_summary, entropy_summary, output_dir)
    create_signature_validation_entropy_plot(performance_summary, signature_entropy_summary, plot_dir)
    create_entropy_change_effect_size_plot(entropy_summary, plot_dir)

    print()
    print("Signature entropy summary:")
    print(signature_entropy_summary.head(10).to_string(index=False))

    print()
    print("Entropy-change effect sizes:")
    print(
        entropy_summary[
            ["model", "layer", "entropy_change_mean", "entropy_change_std", "n_seeds"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
