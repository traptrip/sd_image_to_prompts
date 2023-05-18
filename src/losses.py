from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_metric_learning import losses


class CosineSimilarity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.criterion = nn.CosineSimilarity(*args, **kwargs)

    def forward(
        self,
        logits: Tensor,
        prompts: Tensor,
        targets: Tensor,
        miner: Any,
    ):
        return -self.criterion(logits, prompts).mean()


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss(*args, **kwargs)

    def forward(
        self,
        logits: Tensor,
        prompts: Tensor,
        targets: Tensor,
        miner: Any,
    ):
        return self.criterion(logits, prompts, targets)


class ArcFaceLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.criterion = losses.ArcFaceLoss(*args, **kwargs)

    def forward(
        self,
        logits: Tensor,
        prompts: Tensor,
        targets: Tensor,
        miner: Any,
    ):
        return self.criterion(logits, targets)


class SubCenterArcFaceLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.criterion = losses.SubCenterArcFaceLoss(*args, **kwargs)

    def forward(
        self,
        logits: Tensor,
        prompts: Tensor,
        targets: Tensor,
        miner: Any,
    ):
        return self.criterion(logits, targets)


class TripletMarginLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.criterion = losses.TripletMarginLoss(*args, **kwargs)

    def forward(
        self,
        logits: Tensor,
        prompts: Tensor,
        targets: Tensor,
        miner: Any,
    ):
        embeddings = torch.cat([logits, prompts])
        labels = torch.arange(logits.size(0))
        labels = torch.cat([labels, labels])
        indices_tuple = miner(embeddings, labels) if miner else None
        return self.criterion(embeddings, labels, indices_tuple=indices_tuple)
