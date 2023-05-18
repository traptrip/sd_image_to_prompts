import torch

from typing import List, Tuple, Union

import timm
import torch
import torch.nn as nn
from omegaconf.listconfig import ListConfig

from src.config import NetConfig


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = (1, 1),
        w: Tuple[float, float] = (1, 1),
    ) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size)
        self.gmp = nn.AdaptiveMaxPool2d(output_size)
        self.w = w

    def forward(self, x):
        return self.w[0] * self.gap(x) + self.w[1] * self.gmp(x)


class GemPool2d(nn.Module):
    def __init__(self, p=3, eps=1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return torch.nn.functional.avg_pool2d(
            x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))
        ).pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


def make_pooling(pooling_name: str, default_pooling: torch.nn.Module):
    pooling_mapping = {
        "default": default_pooling,
        "none": torch.nn.Identity(),
        "max": torch.nn.AdaptiveMaxPool2d(1),
        "avg": torch.nn.AdaptiveAvgPool2d(1),
        "avgmax": AdaptiveAvgMaxPool2d(1),
        "gem": GemPool2d(),
    }
    assert (
        pooling_name in pooling_mapping.keys()
    ), f"No such pooling: {pooling_name}, choose one of {list(pooling_mapping.keys())}"

    return pooling_mapping[pooling_name]


def get_backbone(
    backbone_name: str, pretrained: bool = False, grad_checkpointing: bool = False
) -> Tuple[nn.Module, nn.Module, int]:
    if backbone_name in timm.list_models():
        backbone = timm.create_model(backbone_name, pretrained)
        if "efficientnet_b3" in backbone_name:
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            out_dims = 1536
        elif "resnet50" in backbone_name:
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            out_dims = 2048
        else:
            classifier = backbone.get_classifier()
            if "levit" in backbone_name:
                out_dims = classifier[0].l.in_features
            else:
                out_dims = classifier.in_features
            backbone.reset_classifier(-1)
        default_pooling = nn.Identity()
    else:
        raise ValueError(f"{backbone_name} is not recognized")

    if grad_checkpointing:
        backbone.set_grad_checkpointing()

    return backbone, default_pooling, out_dims


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        pooling: nn.Module = nn.Identity(),
        norm_features: bool = False,
        with_projection: bool = True,
        projection_dims: Union[int, List[int]] = 512,
        projection_normalization_layer: str = "none",
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.standardize = (
            nn.LayerNorm(input_dim, elementwise_affine=False)
            if norm_features
            else nn.Identity()
        )
        self.fc = (
            self._create_projection_head(
                input_dim, projection_dims, projection_normalization_layer
            )
            if with_projection
            else nn.Identity()
        )

    def _create_projection_head(
        self,
        input_dim: int = 2048,
        projection_dims: Union[int, List[int]] = 512,
        projection_normalization_layer="none",
    ) -> nn.Module:
        if isinstance(projection_dims, int):
            return nn.Linear(input_dim, projection_dims)

        elif isinstance(projection_dims, (tuple, list)):
            layers = []
            prev_dim = input_dim
            for i, dim in enumerate(projection_dims):
                layers.append(nn.Linear(prev_dim, dim))
                prev_dim = dim
                if i < len(projection_dims) - 1:
                    if projection_normalization_layer == "bn":
                        layers.append(nn.BatchNorm1d(dim))
                    elif projection_normalization_layer == "ln":
                        layers.append(nn.LayerNorm(dim))
                    elif projection_normalization_layer == "none":
                        pass
                    else:
                        raise ValueError(
                            f"Unknown normalization layer : {projection_normalization_layer}"
                        )
                    layers.append(nn.GELU())
            return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.standardize(x)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(
        self,
        net_config: NetConfig,
    ) -> None:
        super().__init__()

        self.backbone, default_pooling, out_dims = get_backbone(
            net_config.backbone_name,
            pretrained=net_config.pretrained,
            grad_checkpointing=net_config.grad_checkpointing,
        )

        self.embedding_size = (
            net_config.projection_dims[-1]
            if isinstance(net_config.projection_dims, list)
            else net_config.projection_dims
        )
        assert (
            net_config.with_projection or self.embedding_size == out_dims
        ), f"Backbone out dimensions number doesn't equal to configured embedding size"

        pooling = make_pooling(net_config.pooling, default_pooling)
        self.projection_head = ProjectionHead(
            out_dims,
            pooling,
            net_config.norm_features,
            net_config.with_projection,
            net_config.projection_dims,
            net_config.projection_normalization_layer,
        )
        self._initialize_weights()

    def get_classifier(self):
        return self.projection_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.projection_head(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def _initialize_weights(self) -> None:
        for param in self.projection_head.parameters():
            if len(param.shape) > 1:
                torch.nn.init.kaiming_normal_(param, mode="fan_in")
            else:
                torch.nn.init.constant_(param, 0)

    def __repr__(self):
        return "ProjectionNet"
