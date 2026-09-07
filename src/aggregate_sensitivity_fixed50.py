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


def load_entropy_histories(log_dir: str) -> pd.DataFrame:
    paths = sorted(Path(log_dir).glob("*_entropy_history.csv"))
    if not paths:
        raise FileNotFoundError(f"No entropy history CSV files found in {log_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_fd_main_history(fd_history_path: str) -> pd.DataFrame:
    path = Path(fd_history_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.aggregate_results first."
        )
    return pd.read_csv(path)


def filter_official_runs(
    df: pd.DataFrame,
    official_seeds: list[int],
    expected_epochs: int,
) -> pd.DataFrame:
    filtered = df[
        df["seed"].isin(official_seeds)
        & (df["epoch"] <= expected_epochs)
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

    if filtered.empty:
        raise ValueError("No complete official sensitivity runs found after filtering.")

    for model in sorted(filtered["model"].unique()):
        model_seeds = sorted([int(seed) for seed in filtered[filtered["model"] == model]["seed"].unique()])
        print(f"{model.upper()} seeds included:", model_seeds)

    return filtered


def summarize_entropy_with_peak(
    df: pd.DataFrame,
    expected_epochs: int,
    binning_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for (model, dataset, seed, layer), group in df.groupby(["model", "dataset", "seed", "layer"]):
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
            "binning_method": binning_method,
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
        raise ValueError(f"No complete groups found for {binning_method}.")

    summary = (
        per_seed
        .groupby(["binning_method", "model", "dataset", "layer"], as_index=False)
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


def summarize_epoch_trajectory(df: pd.DataFrame, binning_method: str) -> pd.DataFrame:
    summary = (
        df
        .groupby(["model", "dataset", "layer", "epoch"], as_index=False)
        .agg(
            activation_entropy_mean=("activation_entropy", "mean"),
            activation_entropy_std=("activation_entropy", "std"),
            train_accuracy_mean=("train_accuracy", "mean"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_generalization_gap_mean=("val_generalization_gap", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary["binning_method"] = binning_method
    return summary


def create_qualitative_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a compact table that checks whether the qualitative direction of entropy
    change is the same across binning methods.
    """
    rows = []

    for (model, layer), group in summary_df.groupby(["model", "layer"]):
        fd = group[group["binning_method"] == "fd"]
        fixed = group[group["binning_method"] == "fixed50"]

        if fd.empty or fixed.empty:
            continue

        fd_change = float(fd["entropy_change_mean"].iloc[0])
        fixed_change = float(fixed["entropy_change_mean"].iloc[0])

        fd_direction = "decrease" if fd_change < 0 else "increase" if fd_change > 0 else "stable"
        fixed_direction = "decrease" if fixed_change < 0 else "increase" if fixed_change > 0 else "stable"

        rows.append({
            "model": model,
            "layer": layer,
            "fd_entropy_change_mean": fd_change,
            "fixed50_entropy_change_mean": fixed_change,
            "fd_direction": fd_direction,
            "fixed50_direction": fixed_direction,
            "same_qualitative_direction": fd_direction == fixed_direction,
            "absolute_change_difference": abs(fd_change - fixed_change),
        })

    return pd.DataFrame(rows)


def plot_fixed50_entropy_curves(epoch_summary: pd.DataFrame, plot_dir: Path) -> None:
    for model in sorted(epoch_summary["model"].unique()):
        model_df = epoch_summary[epoch_summary["model"] == model]

        plt.figure()

        for layer in sorted(model_df["layer"].unique()):
            df = model_df[model_df["layer"] == layer].sort_values("epoch")
            x = df["epoch"]
            y = df["activation_entropy_mean"]
            y_std = df["activation_entropy_std"].fillna(0)

            plt.plot(x, y, label=layer)
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel("Activation Entropy")
        plt.title(f"{model.upper()} Fixed-50 Activation Entropy over Training")
        plt.legend()
        plt.grid(True)

        path = plot_dir / f"fixed50_{model}_activation_entropy_mean_std.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {path}")


def plot_fd_vs_fixed_trajectories(combined_epoch_summary: pd.DataFrame, plot_dir: Path) -> None:
    for model in sorted(combined_epoch_summary["model"].unique()):
        model_df = combined_epoch_summary[combined_epoch_summary["model"] == model]

        plt.figure()

        for method in ["fd", "fixed50"]:
            method_df = model_df[model_df["binning_method"] == method]
            linestyle = "-" if method == "fd" else "--"

            for layer in sorted(method_df["layer"].unique()):
                df = method_df[method_df["layer"] == layer].sort_values("epoch")
                label = f"{layer} {method}"
                plt.plot(df["epoch"], df["activation_entropy_mean"], linestyle=linestyle, label=label)

        plt.xlabel("Epoch")
        plt.ylabel("Activation Entropy")
        plt.title(f"{model.upper()} FD vs Fixed-50 Entropy Trajectories")
        plt.legend()
        plt.grid(True)

        path = plot_dir / f"fd_vs_fixed50_{model}_entropy_trajectories.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {path}")


def plot_entropy_change_comparison(comparison_df: pd.DataFrame, plot_dir: Path) -> None:
    plot_df = comparison_df.copy()
    plot_df["label"] = plot_df["model"].str.upper() + " " + plot_df["layer"]

    x_positions = range(len(plot_df))
    width = 0.35

    plt.figure()
    plt.bar(
        [x - width / 2 for x in x_positions],
        plot_df["fd_entropy_change_mean"],
        width=width,
        label="FD",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        plot_df["fixed50_entropy_change_mean"],
        width=width,
        label="Fixed 50",
    )

    plt.xticks(list(x_positions), plot_df["label"], rotation=45, ha="right")
    plt.xlabel("Model and layer")
    plt.ylabel("Mean entropy change")
    plt.title("FD vs Fixed-50 Entropy Change")
    plt.legend()
    plt.grid(True)

    path = plot_dir / "fd_vs_fixed50_entropy_change_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-log-dir", type=str, default="./results/sensitivity_fixed50/logs")
    parser.add_argument("--fd-history-path", type=str, default="./results/aggregated/combined_entropy_history.csv")
    parser.add_argument("--output-dir", type=str, default="./results/sensitivity_fixed50/aggregated")
    parser.add_argument("--plot-dir", type=str, default="./results/sensitivity_fixed50/plots")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--expected-epochs", type=int, default=30)

    args = parser.parse_args()

    official_seeds = parse_seed_list(args.seeds)
    output_dir = ensure_dir(args.output_dir)
    plot_dir = ensure_dir(args.plot_dir)

    print("Loading fixed-50 sensitivity histories")
    fixed_raw = load_entropy_histories(args.fixed_log_dir)
    fixed_df = filter_official_runs(fixed_raw, official_seeds, args.expected_epochs)

    print("Loading main FD histories")
    fd_raw = load_fd_main_history(args.fd_history_path)
    fd_df = filter_official_runs(fd_raw, official_seeds, args.expected_epochs)

    fixed_per_seed, fixed_summary = summarize_entropy_with_peak(
        fixed_df,
        expected_epochs=args.expected_epochs,
        binning_method="fixed50",
    )
    fd_per_seed, fd_summary = summarize_entropy_with_peak(
        fd_df,
        expected_epochs=args.expected_epochs,
        binning_method="fd",
    )

    fixed_epoch_summary = summarize_epoch_trajectory(fixed_df, binning_method="fixed50")
    fd_epoch_summary = summarize_epoch_trajectory(fd_df, binning_method="fd")

    combined_summary = pd.concat([fd_summary, fixed_summary], ignore_index=True)
    combined_epoch_summary = pd.concat([fd_epoch_summary, fixed_epoch_summary], ignore_index=True)
    comparison = create_qualitative_comparison(combined_summary)

    fixed_df.to_csv(output_dir / "fixed50_combined_entropy_history.csv", index=False)
    fixed_per_seed.to_csv(output_dir / "fixed50_entropy_summary_per_seed.csv", index=False)
    fixed_summary.to_csv(output_dir / "fixed50_entropy_summary_with_peak.csv", index=False)
    combined_summary.to_csv(output_dir / "fd_vs_fixed50_entropy_summary.csv", index=False)
    combined_epoch_summary.to_csv(output_dir / "fd_vs_fixed50_epoch_summary.csv", index=False)
    comparison.to_csv(output_dir / "fd_vs_fixed50_qualitative_comparison.csv", index=False)

    print(f"Saved table: {output_dir / 'fixed50_combined_entropy_history.csv'}")
    print(f"Saved table: {output_dir / 'fixed50_entropy_summary_per_seed.csv'}")
    print(f"Saved table: {output_dir / 'fixed50_entropy_summary_with_peak.csv'}")
    print(f"Saved table: {output_dir / 'fd_vs_fixed50_entropy_summary.csv'}")
    print(f"Saved table: {output_dir / 'fd_vs_fixed50_epoch_summary.csv'}")
    print(f"Saved table: {output_dir / 'fd_vs_fixed50_qualitative_comparison.csv'}")

    plot_fixed50_entropy_curves(fixed_epoch_summary, plot_dir)
    plot_fd_vs_fixed_trajectories(combined_epoch_summary, plot_dir)
    plot_entropy_change_comparison(comparison, plot_dir)

    print()
    print("Fixed-50 entropy summary:")
    print(fixed_summary.to_string(index=False))

    print()
    print("FD vs Fixed-50 qualitative comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
