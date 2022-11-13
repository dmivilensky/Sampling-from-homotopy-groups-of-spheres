import torch


class NormHead(torch.nn.Module):
    def __init__(
        self, dimension=256
    ):
        super().__init__()
        pass

    def forward(self, text, text_lengths):
        raise NotImplementedError()
