import math
import torch
import logging
import torch.nn as nn

from abc import abstractmethod
from typing import Callable
from spikingjelly.activation_based import surrogate, base, neuron

try:
    import cupy
    from spikingjelly.activation_based import neuron_kernel
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cuda_utils = None


class AQIFNode(neuron.BaseNode):
    def __init__(self, v_c: float = 0.8, v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False, init_tau: float = 2.0):
        # self.v_seq = None
        # self.v = None
        assert isinstance(init_tau, float) and init_tau > 1., "Initial tau must be a float greater than 1."
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.v_c = v_c
        self.v_rest = v_rest

        # Initialize w such that tau = init_tau
        init_w = -math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return 'torch', 'cupy'
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', v_c={self.v_c}, tau={tau}, v_rest={self.v_rest}'

    def neuronal_charge(self, x: torch.Tensor):
        """
        Update membrane potential using AQIF dynamics.

        :param x: Input current tensor
        :type x: torch.Tensor
        """
        tau_inv = self.w.sigmoid()
        self.v = self.v + (x + (self.v - self.v_rest) * (self.v - self.v_c)) * tau_inv

    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        Multistep forward propagation for AQIFNode.

        :param x_seq: Input tensor of shape [T, N, *]
        :type x_seq: torch.Tensor
        :return: Output spikes
        :rtype: torch.Tensor
        """
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepQIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.v_reset, self.v_rest,
                self.v_c, self.w.sigmoid(), self.detach_reset, self.surrogate_function.cuda_code
            )

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)
