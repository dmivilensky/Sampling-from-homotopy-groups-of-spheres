import math
from bisect import bisect_right
from typing import List
import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    from https://github.com/javey-q/softcup/
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.999,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor *
            self.gamma ** self.last_epoch
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
