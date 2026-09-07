from collections import OrderedDict
from typing import Dict, Iterable

import torch
from torch import nn


class ActivationRecorder:
    """
    Utility class for recording intermediate activations with PyTorch forward hooks.

    The recorder attaches hooks to selected module names. During a forward pass,
    the latest output of each selected layer is detached, moved to CPU, and stored
    in the activations dictionary.

    This is intended for hidden post-activation layers such as ReLU modules.
    """

    def __init__(self, model: nn.Module, layer_names: Iterable[str]):
        self.model = model
        self.layer_names = list(layer_names)
        self.activations: Dict[str, torch.Tensor] = OrderedDict()
        self.handles = []

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, module_input, module_output):
            if isinstance(module_output, torch.Tensor):
                self.activations[layer_name] = module_output.detach().cpu()
            else:
                raise TypeError(
                    f"Expected tensor output from layer '{layer_name}', "
                    f"but got {type(module_output)}."
                )

        return hook

    def register(self) -> None:
        """Register hooks on the configured layer names."""
        named_modules = dict(self.model.named_modules())

        missing_layers = [
            layer_name for layer_name in self.layer_names
            if layer_name not in named_modules
        ]

        if missing_layers:
            available = ", ".join(named_modules.keys())
            raise ValueError(
                f"Could not find layers: {missing_layers}. "
                f"Available modules are: {available}"
            )

        self.remove()
        self.clear()

        for layer_name in self.layer_names:
            module = named_modules[layer_name]
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self.handles.append(handle)

    def clear(self) -> None:
        """Clear currently stored activations."""
        self.activations.clear()

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return the currently stored activations."""
        return self.activations

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
