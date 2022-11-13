import torch


class EuclideanHead(torch.nn.Module):
    def __init__(
        self, lp=2, **kwargs
    ):
        super().__init__()
        self.lp = lp

    def forward(self, embeddings):
        return torch.cdist(embeddings, embeddings, p=self.lp).unsqueeze(0)
