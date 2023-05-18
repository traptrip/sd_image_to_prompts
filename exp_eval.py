import os
import torch
from pathlib import Path
from scipy.spatial import distance
import timm
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from torchvision import transforms

from src.training import Trainer
from src.lion_optimizer import Lion
from src.losses import CosineEmbeddingLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_LR = 1e-4


class NetConfig:
    backbone_name = "vit_huge_patch14_224_clip_laion2b"
    pooling = "none"
    norm_features = False
    with_projection = True
    projection_dims = 384
    projection_normalization_layer = "none"
    pretrained = True
    grad_checkpointing = True


class Config:
    tg_token = None
    tg_chat_id = None

    n_epochs = 5
    n_warm_epochs = 0
    n_warm_steps = 0
    batch_size = 512  # 112 512 64
    n_workers = 4
    use_amp = False
    clip_value = None
    # backbone_lr = 1e-4
    # head_lr = 1e-3
    backbone_lr = 3e-5
    head_lr = 3e-4

    # data
    data_dir = Path("./data/test_data/")
    # metadata_filepath = data_dir / "metadata.csv"
    metadata_filepath = Path("./data/test_data/metadata.csv")

    # exp paths
    resume = False
    exp_dir = Path("runs/sd_min_step_1__vit_huge__m0.5__negaug_0.1__adamw")
    pretrained_weights = None

    # training params
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = NetConfig()
    criterion = None
    optimizer = None
    scheduler = None
    miner = None

    transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(size=[224, 224]),
                # transforms.RandomResizedCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=[224, 224]),
                # transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        ),
    }


cfg = Config()
net: torch.nn.Module = timm.create_model(
    cfg.net.backbone_name, cfg.net.pretrained, num_classes=cfg.net.projection_dims
)
net.to(cfg.device)

cfg.net = net
cfg.criterion = CosineEmbeddingLoss(0.5, None)


trainer = Trainer(cfg)
trainer.eval()
