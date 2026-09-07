import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility within a fixed environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seeds DataLoader workers when num_workers > 0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Returns the CPU device used for the experiment."""
    return torch.device("cpu")
