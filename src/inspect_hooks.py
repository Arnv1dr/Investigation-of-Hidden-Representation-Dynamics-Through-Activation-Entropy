import argparse

import torch

import config
from src.data import get_dataloaders
from src.hooks import ActivationRecorder
from src.models import create_model, get_monitored_layer_names
from src.utils import get_device, set_seed


def inspect_activation_hooks(
    model_name: str,
    dataset_name: str,
    batch_size: int,
    seed: int,
    split_seed: int,
    val_per_class: int,
) -> None:
    """
    Verifies that activation hooks correctly record hidden post-activation layers.

    This script runs a single validation batch through the model and prints the
    activation shape of every monitored layer.
    """
    set_seed(seed)
    device = get_device()

    _, val_loader, _ = get_dataloaders(
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
    model.eval()

    layer_names = get_monitored_layer_names(model_name)

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {batch_size}")
    print(f"Monitored hidden post-activation layers: {layer_names}")

    images, labels = next(iter(val_loader))
    images = images.to(device)

    recorder = ActivationRecorder(model=model, layer_names=layer_names)
    recorder.register()

    recorder.clear()

    with torch.no_grad():
        outputs = model(images)

    print(f"Input batch shape: {tuple(images.shape)}")
    print(f"Output logits shape: {tuple(outputs.shape)}")
    print()
    print("Recorded activation shapes:")

    activations = recorder.get_activations()

    for layer_name in layer_names:
        activation = activations.get(layer_name)

        if activation is None:
            print(f"- {layer_name}: NOT RECORDED")
        else:
            print(f"- {layer_name}: {tuple(activation.shape)}")

    recorder.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--dataset", type=str, default="FashionMNIST", choices=["MNIST", "FashionMNIST"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--split-seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--val-per-class", type=int, default=config.VAL_PER_CLASS)

    args = parser.parse_args()

    inspect_activation_hooks(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        split_seed=args.split_seed,
        val_per_class=args.val_per_class,
    )


if __name__ == "__main__":
    main()
