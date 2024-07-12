import logging
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from util import data_utils, model_utils

logger = logging.getLogger()
LR = 0.003  # Learning rate
N = 100  # data len
K = 1  # number of client
index = 0  # index of current client


def register_hooks(model):
    activations = {}

    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        layer_name = class_name + "_" + str(len(activations))
        activations[layer_name] = output

    for name, layer in model.named_children():
        layer.register_forward_hook(hook_fn)

    return activations


def calculate_activation_sizes(activations):
    total_size = 0
    for layer_name, activation in activations.items():
        activation_size = activation.numel() * activation.element_size()  # Number of elements * bytes per element
        total_size += activation_size
        print(f"{layer_name} activation size: {activation_size / (1024 * 1024):.3f} MB")
    print(f"Total activation size: {total_size / (1024 * 1024):.3f} MB")


def train_model():
    cpu_count = multiprocessing.cpu_count()
    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]

    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is using...")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"MPS is not available. Using {device}")

    trainloader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, cpu_count)

    splitPoint = [6, 6]
    net: nn.Module = model_utils.get_model('Client', splitPoint, 'cpu', True)
    logger.debug(net)
    criterion = nn.CrossEntropyLoss()
    activations = register_hooks(net)

    if len(list(net.parameters())) != 0:
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    net.to(device)
    net.train()
    for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    layer_sizes = {}
    for name, param in net.named_parameters():
        if param.requires_grad:
            layer_name = name.split('.')[0]  # Extract the layer name (e.g., 'conv1', 'fc1')
            param_size = param.numel() * param.element_size()  # Number of elements * bytes per element
            if layer_name in layer_sizes:
                layer_sizes[layer_name] += param_size
            else:
                layer_sizes[layer_name] = param_size
    print(f"Backward Propagation data size:\n")
    for layer, size in layer_sizes.items():
        print(f"{layer} gradient size: {size / (1024 * 1024):.3f} MB")
    print(f"Total gradient size: {sum(layer_sizes.values()) / (1024 * 1024):.3f} MB")

    print(f"Forward Propagation data size:\n")
    calculate_activation_sizes(activations)

