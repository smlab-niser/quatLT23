import torch
from torch import nn
import torch.nn.utils.prune as prune

from htorch import layers


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
    if fraction==0:
        return model
    if fraction < 0 or fraction > 1:
        raise ValueError("Pruning fraction should be in [0, 1]")

    # Iterate through each module in the model and apply pruning
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune_method(module, name='weight', amount=fraction)
        elif isinstance(module, layers.QConv2d) or isinstance(module, layers.QLinear):
            prune_method(module, name='r_weight', amount=fraction)
            prune_method(module, name='i_weight', amount=fraction)
            prune_method(module, name='j_weight', amount=fraction)
            prune_method(module, name='k_weight', amount=fraction)

    return model


def reset_model(m):
    """
    Reset the weights of the model's learnable parameters using the Xavier uniform initialization.

    Args:
        m (torch.nn.Module): The module whose weights need to be reset.

    Examples:
        # Reset the weights of a model
        model = MyModel()
        torch.manual_seed(seed)
        model.apply(reset_model)
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        try:torch.nn.init.xavier_uniform_(m.weight_orig.data)
        except:torch.nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, layers.QConv2d) or isinstance(m, layers.QLinear):
        try:
            nn.init.xavier_uniform_(m.r_weight_orig.data)
            nn.init.xavier_uniform_(m.i_weight_orig.data)
            nn.init.xavier_uniform_(m.j_weight_orig.data)
            nn.init.xavier_uniform_(m.k_weight_orig.data)
        except:
            nn.init.xavier_uniform_(m.r_weight.data)
            nn.init.xavier_uniform_(m.i_weight.data)
            nn.init.xavier_uniform_(m.j_weight.data)
            nn.init.xavier_uniform_(m.k_weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        try:
            nn.init.constant_(m.weight_orig.data, 1)
            nn.init.constant_(m.bias_orig.data, 0)
        except:
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)


def get_prune_percentage(model):
    """
    Computes the percentage of weights pruned from a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to be pruned.

    Returns:
        float: Part of weights remaining in the model.
    """
    total_weights = 0
    remaining_weights = 0

    # Iterate through each module in the model and compute the percentage of weights pruned
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            total_weights += module.weight.numel()
            try: remaining_weights += module.weight_mask.sum().item()
            except: remaining_weights += module.weight.numel()
        elif isinstance(module, layers.QConv2d) or isinstance(module, layers.QLinear):
            total_weights += module.r_weight.numel()
            total_weights += module.i_weight.numel()
            total_weights += module.j_weight.numel()
            total_weights += module.k_weight.numel()
            try:
                remaining_weights += module.r_weight_mask.sum().item()
                remaining_weights += module.i_weight_mask.sum().item()
                remaining_weights += module.j_weight_mask.sum().item()
                remaining_weights += module.k_weight_mask.sum().item()
            except:
                remaining_weights += module.r_weight.numel()
                remaining_weights += module.i_weight.numel()
                remaining_weights += module.j_weight.numel()
                remaining_weights += module.k_weight.numel()

    return remaining_weights / total_weights


def number_of_parameters(model):
    """Counts the number of parameters given a model.

    Args:
        model: PyTorch model to be pruned.
    
    Returns:
        int: Number of parameters in the model.
    """
    total_weights = 0

    # Iterate through each module in the model and compute the number of parameters
    for module_name, module in model.named_modules():
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.BatchNorm2d)
            or isinstance(module, layers.QConv2d) 
            or isinstance(module, layers.QLinear)
        ):
            for param in module.parameters():
                total_weights += param.numel()

    return total_weights