import argparse
import collections
import logging
import os.path
import urllib.request

import numpy as np
import torch
import torch.nn.init as init
import tqdm

from models.nn_model.vgg5 import *

fed_logger = logging.getLogger()

np.random.seed(0)
torch.manual_seed(0)

MODEL_BASE_DIR = 'models.nn_model.'
MODEL_NAME = 'vgg5'


def get_model(location, layer, device, edge_based):
    net = get_class()()
    net = net.initialize(location, layer, edge_based)
    net = net.to(device)
    fed_logger.debug(str(net))
    return net


def split_weights_client(weights, cweights):
    """
    evaluate client weights
    """
    for key in cweights:
        assert cweights[key].size() == weights[key].size()
        cweights[key] = weights[key]
    return cweights


def split_weights_server(weights, cweights, sweights, eweights):
    """
    evaluate server weights
    """
    ckeys = list(cweights)
    skeys = list(sweights)
    ekeys = list(eweights)
    keys = list(weights)

    for i in range(len(skeys)):
        assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys) + len(ekeys)]].size()
        sweights[skeys[i]] = weights[keys[i + len(ckeys) + len(ekeys)]]

    return sweights


def split_weights_edgeserver(weights, cweights, eweights):
    ckeys = list(cweights)
    ekeys = list(eweights)
    keys = list(weights)

    for i in range(len(ekeys)):
        assert eweights[ekeys[i]].size() == weights[keys[i + len(ckeys)]].size()
        eweights[ekeys[i]] = weights[keys[i + len(ckeys)]]

    return eweights


def concat_weights(weights, cweights, sweights):
    concat_dict = collections.OrderedDict()

    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)

    for i in range(len(ckeys)):
        concat_dict[keys[i]] = cweights[ckeys[i]]

    for i in range(len(skeys)):
        concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

    return concat_dict


def zero_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.zeros_(m.weight)
            init.zeros_(m.bias)
            init.zeros_(m.running_mean)
            init.zeros_(m.running_var)
        elif isinstance(m, nn.Linear):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    return net


def norm_list(alist):
    return [l / sum(alist) for l in alist]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test(uninet, testloader, device, criterion):
    uninet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = uninet(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    fed_logger.info('Test Accuracy: {}'.format(acc))

    # Save checkpoint.
    torch.save(uninet.state_dict(), './' + MODEL_NAME + '.pth')

    return acc


def get_class():
    kls = MODEL_BASE_DIR + MODEL_NAME + '.' + MODEL_NAME
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def download_model(link):
    if not link == '':
        if not check_link_integrity():
            urllib.request.urlretrieve(link, MODEL_BASE_DIR)
            if not check_link_integrity():
                fed_logger.error('the file is not downloaded correctly or its name doesn\'t follow the proper format')


def check_link_integrity():
    return os.path.exists(MODEL_BASE_DIR + MODEL_NAME)


def get_unit_model_len():
    return len(get_class()().get_config())


def get_unit_model() -> NNModel:
    return get_class()()
