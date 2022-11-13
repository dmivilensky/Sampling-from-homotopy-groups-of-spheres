import torch
import numpy


class HyperbolicHead(torch.nn.Module):
    def __init__(
        self, method="poincare", **kwargs
    ):
        super().__init__()
        self.method = method

    @staticmethod
    def __proj(x, eps=1e-5):
        norm = torch.linalg.norm(x, axis=-1).unsqueeze(-1)
        return torch.where(norm >= 1, x / norm - eps, x)

    @staticmethod
    def __dist(u, v):
        u1 = HyperbolicHead.__proj(u)
        v1 = HyperbolicHead.__proj(v)
        return torch.acosh(torch.linalg.norm(u1 - v1, axis=-1) /
                        (1 - torch.linalg.norm(u1, axis=-1)**2) /
                        (1 - torch.linalg.norm(v1, axis=-1)**2) * 2 + 1)

    def forward(self, embeddings):
        pairwise = self.__dist(
            embeddings[..., None, :, :], embeddings[..., :, None, :])
        return pairwise.unsqueeze(0)
