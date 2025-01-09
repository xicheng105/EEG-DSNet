import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil
from spikingjelly.activation_based import neuron, surrogate, layer, functional

from UtiliTies import SeparableConv2D, Permute, IterateSeparableConv2D, DilatedConv2d

import Neuron


# %% EEG_DSNet
class EEG_DSNet(nn.Module):
    def __init__(self, T: int, neuron_type: str, num_classes=2, input_channels=22, input_length=1000,
                 virtual_channels=8, kernel_size=128, dilation_layers=2, dilation_rate=2, dilation_kernel_size=6):
        super(EEG_DSNet, self).__init__()

        self.T = T
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_length = input_length
        self.virtual_channels = virtual_channels
        self.kernel_size = kernel_size
        self.dilation_layers = dilation_layers
        self.dilation_rate = dilation_rate
        self.dilation_kernel_size = dilation_kernel_size
        self.neuron_type = neuron_type

        self.TemporalConvBlock = self._conv_block(virtual_channels=self.virtual_channels, TemporalBranch=True)
        self.SpectralConvBlock = self._conv_block(virtual_channels=self.virtual_channels * 2, TemporalBranch=False)
        self.FCBlock = self._fc_block()

        functional.set_step_mode(self, step_mode='m')

    def spiking_neuron(self):
        if self.neuron_type == 'IFNode':
            return neuron.IFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'LIFNode':
            return neuron.LIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'ParametricLIFNode':
            return neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'QIFNode':
            return neuron.QIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'EIFNode':
            return neuron.EIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'IzhikevichNode':
            return neuron.IzhikevichNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'LIAFNode':
            return neuron.LIAFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'KLIFNode':
            return neuron.KLIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'AQIFNode':
            return Neuron.AQIFNode(surrogate_function=surrogate.ATan())
        else:
            raise ValueError(f"Unsupported neuron type: {self.neuron_type}")

    def _conv_block(self, virtual_channels, TemporalBranch=True):
        layers = [
            nn.ZeroPad2d(((self.kernel_size // 2 - 1), self.kernel_size // 2, 0, 0)),
            layer.Conv2d(
                in_channels=1,
                out_channels=virtual_channels,
                kernel_size=(1, self.kernel_size),
                bias=False
            ),
            # Temporal: (virtual_channels, input_channels, input_length)
            # Spectral: (virtual_channels * 2, input_channels, input_length)
            layer.BatchNorm2d(virtual_channels),
            self.spiking_neuron(),
            layer.Conv2d(
                in_channels=virtual_channels,
                out_channels=2 * virtual_channels,
                kernel_size=(self.input_channels, 1),
                groups=virtual_channels,
                bias=False
            ),
            # Temporal: (virtual_channels * 2, 1, input_length)
            # Spectral: (virtual_channels * 4, 1, input_length)
            layer.BatchNorm2d(2 * virtual_channels),
            self.spiking_neuron(),
        ]
        if TemporalBranch:
            layers.append(layer.AvgPool2d(kernel_size=(1, self.kernel_size // 32)))
            # (virtual_channels * 2, 1, input_length // 4)
        else:
            layers.append(layer.MaxPool2d(kernel_size=(1, self.kernel_size // 32)))
            # (virtual_channels * 4, 1, input_length // 4)
        layers += [
            SeparableConv2D(
                in_channels=2 * virtual_channels,
                out_channels=2 * virtual_channels,
                kernel_size=(1, self.kernel_size // 4),
                spiking=True
            ),
            # Temporal: (virtual_channels * 2, 1, input_length // 4)
            # Spectral: (virtual_channels * 4, 1, input_length // 4)
            layer.BatchNorm2d(2 * virtual_channels),
            self.spiking_neuron()
        ]
        if TemporalBranch:
            layers.append(layer.AvgPool2d(kernel_size=(1, self.kernel_size // 16), ceil_mode=True))
            # (virtual_channels * 2, 1, ceil(T / 32))
        else:
            layers.append(layer.MaxPool2d(kernel_size=(1, self.kernel_size // 8), ceil_mode=True))
            # (virtual_channels * 4, 1, ceil(T / 64))

        if not TemporalBranch:  # (ceil(T / 64), 1, virtual_channels * 4)
            layers.append(Permute(0, 1, 4, 3, 2))
            # (ceil(T / 64), 1, virtual_channels * 4)
        layers += [
            SeparableConv2D(
                in_channels=2 * self.virtual_channels,
                out_channels=4 * self.virtual_channels,
                kernel_size=(1, self.dilation_kernel_size),
                dilation_rate=(1, self.dilation_rate),
                spiking=True
            ),
            # self.spiking_neuron(),
            SeparableConv2D(
                in_channels=4 * self.virtual_channels,
                out_channels=2 * self.virtual_channels,
                kernel_size=(1, self.dilation_kernel_size),
                dilation_rate=(1, self.dilation_rate + 1),
                spiking=True
            ),
            layer.BatchNorm2d(2 * self.virtual_channels),
            self.spiking_neuron(),
            # Temporal: (2 * virtual_channels, 1, ceil(T / 32))
            # Spectral: (ceil(T / 64), 1, virtual_channels * 4)
            layer.Flatten()
        ]
        return nn.Sequential(*layers)

    def _fc_block(self):
        layers = [
            layer.Linear(in_features=512 + 512, out_features=self.num_classes, bias=False),
            self.spiking_neuron()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batches, virtual_channels, channels, length)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, batches, virtual_channels, channels, length)
        BranchTx = self.TemporalConvBlock(x_seq)  # 512
        # print(f"Shape of BranchTx: {BranchTx.shape}")
        BranchSx = self.SpectralConvBlock(x_seq)  # 512
        # print(f"Shape of BranchSx: {BranchSx.shape}")
        Concat = torch.cat((BranchTx, BranchSx), dim=2)
        FC = self.FCBlock(Concat)
        # return Concat
        fr = FC.mean(0)
        return F.softmax(fr, dim=1)

    def spiking_encoder(self):
        return self._conv_block[1:]
