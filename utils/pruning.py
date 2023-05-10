import torch
from torch import nn
import torch.nn.utils.prune as prune


def prune_model(model, fraction):
    """
    Prunes the weights of a PyTorch model based on the given pruning fraction.

    Args:
        model (nn.Module): PyTorch model to be pruned.
        fraction (float): Fraction of weights to be pruned. Value should be between 0 and 1.

    Returns:
        nn.Module: Pruned PyTorch model with trainable weights.
    """
    # Identify the pruning method to be used based on the model type
    if isinstance(model, nn.Module):
        prune_method = prune.l1_unstructured
    else:
        raise ValueError("The provided model is not a valid nn.Module.")

    # Validate input fraction
    if fraction==1:
        return model
    if fraction <= 0 or fraction > 1:
        raise ValueError("Pruning fraction should be in (0, 1]")

    # Iterate through each module in the model and apply pruning
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune_method(module, 'weight', amount=fraction)

    # Remove the pruned weights
    # prune.remove(model, 'weight')

    return model