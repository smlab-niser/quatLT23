import os
import pathlib
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from htorch import utils


def dataset_dir():
    """
    Specify where to save the datasets.
    """
    return os.path.join(pathlib.Path.home(), 'Documents', 'datasets')


def results_dir():
    """
    Specify where to save the results.
    """
    return os.path.join(pathlib.Path.home(), 'Documents', 'results')


def _load_from_set(trainset, testset, batch_size, model_to_run, collate=True):
    """
    Loads data with an additional channel for quat models if necessary.
    If collate=True, adds grayscale version as the fourth
    channel to input images.
    """
    collate_fn = None
    if collate and model_to_run == 'quat':
        collate_fn = utils.convert_data_for_quaternion

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,   # was 2 diptarko told to change to 8
        # pin_memory=True,    # new added
        collate_fn=collate_fn
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # was 2 diptarko told to change to 8
        # pin_memory=True,   # new added
        collate_fn=collate_fn
    )

    return trainloader, testloader


def data_loader(model_to_run, dataset, batch_size):
    """
    Data loader function.
    Has different conditions for different datasets.
    """
    data_directory = os.path.join(dataset_dir(), dataset)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_directory,
            train=True,
            download=True,
            transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_directory,
            train=False,
            download=True,
            transform=test_transform
        )

        trainloader, testloader = _load_from_set(
            trainset, testset, batch_size, model_to_run)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=data_directory,
            train=True,
            download=True,
            transform=train_transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_directory,
            train=False,
            download=True,
            transform=test_transform
        )

        trainloader, testloader = _load_from_set(
            trainset, testset, batch_size, model_to_run)

    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        trainset = torchvision.datasets.MNIST(
            root=data_directory,
            train=True,
            download=True,
            transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=data_directory,
            train=False,
            download=True,
            transform=transform
        )

        trainloader, testloader = _load_from_set(
            trainset, testset, batch_size, model_to_run, collate=False)

    else:
        raise ValueError("Invalid dataset.")

    return trainloader, testloader


def format_text(input, length=40, heading=True):
    """
    Input has to be a string with even length.
    """
    input_len = len(input)
    if input_len % 2 != 0:
        raise ValueError("The input is not of even length.")
    else:
        num_dashes = int((length - len(input) - 4) / 2)
        if heading:
            return ('-'*length + '\n' + '-'*num_dashes + '  ' + input
                    + '  ' + '-'*num_dashes + '\n' + '-'*length + '\n')
        else:
            return ('-'*num_dashes + '  ' + input + '  '
                    + '-'*num_dashes + '\n')


def display_model(model: nn.Module, output_directory=None, show=True):
    """
    Prints the structure of the neural network
    model (its constituent layers and the
    number of parameters in each) as a table.
    """
    table = PrettyTable(["Layers", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    table.add_row(["Total trainable parameters", total_params])

    if show:
        print(format_text("Model statistics"))
        print(table)
        print('\n')

    if output_directory:
        file_path = os.path.join(output_directory, 'model_structure.txt')
        with open(file_path, 'w') as file:
            file.write(str(table))


def get_trainable_params(model: nn.Module):
    """
    Returns the number of trainable parameters in the network.
    """
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_param


def qr_relative_sparsity(quat: nn.Module, real: nn.Module):
    """
    Function to get the relative sparsity of quat and real models.
    """
    q_params = float(get_trainable_params(quat))
    r_params = float(get_trainable_params(real))
    return q_params/r_params
