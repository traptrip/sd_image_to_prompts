import torch
from pathlib import Path
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

DEFAULT_LR = 1e-4


class NetConfig:
    backbone_name = "vit_large_patch14_224_clip_laion2b"
    pooling = "none"
    norm_features = False
    with_projection = True
    projection_dims = 384
    projection_normalization_layer = "none"
    pretrained = True
    grad_checkpointing = False


class CriterionConfig:
    f = torch.nn.CosineEmbeddingLoss
    args = dict(margin=0, size_average=None)


class OptimizerConfig:
    f = torch.optim.Adam
    args = dict(lr=DEFAULT_LR)


class SchedulerConfig:
    f = torch.optim.lr_scheduler.CosineAnnealingLR
    args = dict(T_max=8, eta_min=DEFAULT_LR * 1e-2, last_epoch=-1, verbose=False)


class Config:
    seed: 42

    # data
    data_dir = Path("data/sd2_data")
    metadata_filepath = data_dir / "metadata.csv"

    # exp paths
    resume = False
    exp_dir = Path("runs/baseline")
    pretrained_weights = None

    # training params
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 10
    n_warm_epochs = 1
    batch_size = 16
    n_workers = 4
    use_amp = True
    clip_value = None

    net = NetConfig()
    criterion = CriterionConfig()
    optimizer = OptimizerConfig()
    scheduler = SchedulerConfig()

    transforms = {
        "train": {
            "Resize": dict(size=[224, 224]),
            "ToTensor": {},
            "Normalize": dict(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        },
        "test": {
            "Resize": dict(size=[232, 232]),
            "CenterCrop": dict(size=[224, 224]),
            "ToTensor": {},
            "Normalize": dict(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        },
    }


cfg = Config()
