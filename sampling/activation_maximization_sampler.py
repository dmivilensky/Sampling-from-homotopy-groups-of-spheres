from typing import Tuple, Dict, List, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange

from pathlib import Path

class LocalNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latent):
        mean, std = latent.mean(dim = 0), latent.std(dim = 0)
        return (latent - mean[None, :]) / std[None, :]


class GlobalNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latent):
        mean, std = latent.mean(), latent.std()
        return (latent - mean) / std


class LocalScaling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.randn(channels))
        self.beta = nn.Parameter(torch.randn(channels))

    def forward(self, latent):
        output = latent * self.gamma[None, :] + self.beta[None, :]
        return output


class GlobalScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, latent):
        output = latent * self.gamma + self.beta
        return output


class SoftmaxSampler(nn.Module):
    def __init__(self, n_samples: int):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x):
        return softmax_sample(x, self.n_samples)


class SoftmaxSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: "torch.Tensor", n_samples: int = 16) -> "torch.Tensor":
        n_classes = input.shape[-1]

        with torch.enable_grad():
            input_clone = input.clone().detach().requires_grad_(True)
            softmaxed = F.softmax(input_clone, dim = -1)
        output = torch.multinomial(softmaxed, num_samples = n_samples, replacement = True).permute(1, 0)
        output = F.one_hot(output, num_classes = n_classes)

        ctx.save_for_backward(input_clone, softmaxed)

        return output.to(torch.float32).requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        input_clone, softmaxed = ctx.saved_tensors
        torch.autograd.backward(softmaxed, grad_output.mean(dim = 0))

        return input_clone.grad, None


class ArgmaxSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: "torch.Tensor", n_samples: int = 16) -> "torch.Tensor":
        n_classes = input.shape[-1]

        output = torch.stack([torch.argmax(input, dim = -1) for _ in range(n_samples)])

        output = F.one_hot(output, num_classes = n_classes)
        return output.to(torch.float64).requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        grad_input = grad_output.clone()
        return grad_input, None


class GumbelSoftmaxSample:

    @staticmethod
    def apply(logits, n_samples, tau = 1.):
        return torch.stack([F.gumbel_softmax(logits, tau = tau, hard = True, dim = -1) for _ in range(n_samples)])


softmax_sample = SoftmaxSample.apply

argmax_sample = ArgmaxSample.apply

gumbel_softmax_sample = GumbelSoftmaxSample.apply


@dataclass
class ActivationMaximizationSampler:
    vocab: Dict[str, int]
    generators_number: int
    
    batch_size: int
    iterations: int
    length: int

    normalization_method: str
    scaling_method: str
    sample_method: str

    classifiers: List['nn.Module']
    
    classifiers_reduction_method: str
    batch_reduction_method: str

    optimizer_config: Tuple[str, Dict[str, Any]]
    
    device: 'torch.device'

    def __post_init__(self):

        norm  = GlobalNormalization() if self.normalization_method == 'global' else LocalNormalization()
        object.__setattr__(self, 'norm', norm)

        scale = GlobalScaling() if self.scaling_method == 'global' else LocalScaling()
        object.__setattr__(self, 'scale', scale)

        sample = SoftmaxSampler(self.batch_size) if self.sample_method == 'softmax' else None
        object.__setattr__(self, 'sample', sample)

        classifier_reduction = lambda x: x.sum(dim = 0) if self.classifier_reduction_method == 'sum' else None
        batch_reduction      = lambda x: x.mean() if self.batch_reduction_method == 'mean' else None
        object.__setattr__(self, 'classifier_reduction', classifier_reduction)
        object.__setattr__(self, 'batch_reduction', batch_reduction)

        name, args = self.optimizer_config
        args['maximize'] = True
        optimizer_factory = lambda parameters: object.__getattribute__(torch, name)(parameters, **args)
        object.__setattr__(self, 'optimizer_factor', optimizer_factory)


    def __optimize__(self):
        z = torch.rand((self.length, len(self.vocab)), device = self.device, requires_grad = True)
        optimizer = self.optimizer_factory((z,))

        for _ in trange(self.iterations):
            input  = self.norm(z)
            input  = self.scale(z)
            input  = self.sample(z)
            output = torch.stack([classifier(input) for classifier in self.classifiers])
            output = self.classifiers_reduction(output)
            output = self.batch_reduction(output)

            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        
        self.batch = self.sample(self.scale(self.norm(z)))
        self.idx = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.idx == len(self.batch_size):
            self.__optimize__()
        result = self.batch[self.idx]
        self.idx += 1
        return result
