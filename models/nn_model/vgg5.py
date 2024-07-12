import torch.nn as nn

from models.interface.nn_model_interface import NNModel


# Build the VGG nn_model according to location and split_layer
class vgg5(NNModel):
    def _make_layers(self, edge_based):
        features = []
        denses = []
        cfg = self.get_config()
        if edge_based:
            if self.location == 'Server':
                cfg = cfg[self.split_layer[1] + 1:]

            if self.location == 'Client':
                cfg = cfg[:self.split_layer[0] + 1]

            if self.location == 'Edge':
                cfg = cfg[self.split_layer[0] + 1:self.split_layer[1] + 1]
        else:
            if self.location == 'Server':
                cfg = cfg[self.split_layer + 1:]

            if self.location == 'Client':
                cfg = cfg[:self.split_layer + 1]

        if self.location == 'Unit':  # Get the holistic nn_model
            pass

        for x in cfg:
            in_channels, out_channels = x[1], x[2]
            kernel_size = x[3]
            if x[0] == 'M':
                features += [nn.MaxPool2d(kernel_size=kernel_size, stride=2)]
            if x[0] == 'D':
                denses += [nn.Linear(in_channels, out_channels)]
            if x[0] == 'C':
                features += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True)]

        return nn.Sequential(*features), nn.Sequential(*denses)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(self.features) > 0:
            out = self.features(x)
        else:
            out = x
        if len(self.denses) > 0:
            out = out.view(out.size(0), -1)
            out = self.denses(out)

        return out

    def get_config(self):
        return [('C', 3, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 3), ('M', 32, 32, 2, 32 * 16 * 16, 0),
                ('C', 32, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 32), ('M', 64, 64, 2, 64 * 8 * 8, 0),
                ('C', 64, 64, 3, 64 * 8 * 8, 64 * 8 * 8 * 3 * 3 * 64),
                ('D', 8 * 8 * 64, 128, 1, 64, 128 * 8 * 8 * 64),
                ('D', 128, 10, 1, 10, 128 * 10)]
