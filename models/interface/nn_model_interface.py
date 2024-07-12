from abc import ABC, abstractmethod
import torch.nn as nn


class NNModel(ABC, nn.Module):
    """
    class name should be the same as file name
    """

    def __init__(self):
        super(NNModel, self).__init__()
        self.split_layer = None
        self.location = None
        self.cfg = self.get_config()
        self.features, self.denses = None, None

    def initialize(self, location, split_layer, edge_based):
        if location == 'Unit':  # Get the holistic nn_model
            self.location = location
            self.features, self.denses = self._make_layers(edge_based)
            self._initialize_weights()
        elif edge_based:
            split_point = split_layer[0]
            assert split_point < len(self.cfg)
            self.split_layer = split_layer
            self.location = location
            self.features, self.denses = self._make_layers(edge_based)
            self._initialize_weights()
        else:
            split_point = split_layer
            assert split_point < len(self.cfg)
            self.split_layer = split_layer
            self.location = location
            self.features, self.denses = self._make_layers(edge_based)
            self._initialize_weights()

        return self

    def forward(self, x):
        if len(self.features) > 0:
            out = self.features(x)
        else:
            out = x
        if len(self.denses) > 0:
            out = out.view(out.size(0), -1)
            out = self.denses(out)

        return out

    @abstractmethod
    def _make_layers(self, edge_based):
        """
        notice that you can change any part of the method if the input and output still matches the requirements
        :param cfg: the configuration of each layer in a list
        :return: nn.Sequential(*features), nn.Sequential(*denses)
        """
        cfg = self.get_config()
        features = []
        denses = []
        if self.location == 'Unit':  # Get the holistic nn_model
            pass

        elif self.location == 'Server':
            cfg = cfg[self.split_layer[1] + 1:]

        elif self.location == 'Client':
            cfg = cfg[:self.split_layer[0] + 1]

        elif self.location == 'Edge':
            cfg = cfg[self.split_layer[0] + 1:self.split_layer[1] + 1]

        #  TODO fill the featured and dense based on your model configuration
        nn.Sequential(*features), nn.Sequential(*denses)

    @abstractmethod
    def _initialize_weights(self):
        # you can refer to implemented vgg class to see an example
        pass

    @abstractmethod
    def get_config(self):
        """

        :return: the configuration of each layer in a list  of (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
        """
        pass
