import os
import torch
from pathlib import Path
from scipy.spatial import distance as scipy_distance
from torchvision import transforms
import timm
from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from pytorch_metric_learning import distances, losses, miners, reducers, testers

from src.training import Trainer
from src.lion_optimizer import Lion
from src import losses, nets

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NetConfig:
    backbone_name = "vit_huge_patch14_224_clip_laion2b"  # "vit_large_patch14_224_clip_laion2b"  # "vit_large_patch14_224_clip_laion2b" levit_384 hf-hub:timm/convnext_base.clip_laion2b_augreg_ft_in12k
    pooling = "none"
    norm_features = False
    with_projection = True
    projection_dims = 384
    projection_normalization_layer = "ln"
    pretrained = True
    grad_checkpointing = True


class Config:
    tg_token = None
    tg_chat_id = None

    n_epochs = 6
    n_warm_epochs = 0
    n_warm_steps = 0
    batch_size = 2  # 112 512 64
    n_workers = 4
    use_amp = True
    clip_value = 1.0

    # backbone_lr = 1e-5
    # head_lr = 1e-4

    backbone_lr = 3e-6
    head_lr = 3e-5

    # data
    data_dir = Path("data/sd_data")
    metadata_filepath = data_dir / "metadata_min_step=1.csv"

    # exp paths
    resume = False
    exp_dir = Path("runs/sd_min_step_1__vit_huge__m0.5__negaug_0.1__adamw")

    pretrained_weights = None
    # pretrained_weights = torch.load(
    #     "runs/sd__vit_large_336__m_0.5__negaug_0.1__adamw/model_2ep.pth",
    #     map_location="cuda",
    # )["net_state"]

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
                # transforms.RandomHorizontalFlip(),
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


if __name__ == "__main__":
    cfg = Config()

    net: torch.nn.Module = timm.create_model(
        cfg.net.backbone_name, cfg.net.pretrained, num_classes=cfg.net.projection_dims
    )
    if cfg.pretrained_weights is not None:
        net.load_state_dict(cfg.pretrained_weights)

    if cfg.net.grad_checkpointing:
        net.set_grad_checkpointing()

    # net = nets.Net(cfg.net)

    net.to(cfg.device)

    ### LOSS FUNC ###
    criterion = losses.CosineEmbeddingLoss(0.5, None)

    # criterion = losses.CosineSimilarity()

    # criterion = losses.SubCenterArcFaceLoss(
    #     num_classes=int(cfg.metadata_filepath.stem.split("_")[-1]),
    #     embedding_size=384,
    #     margin=16,
    #     scale=30,
    #     sub_centers=10,
    # )

    # margin = 0.5
    # distance = distances.CosineSimilarity()
    # reducer = reducers.ThresholdReducer(low=0)
    # mining_func = miners.TripletMarginMiner(
    #     margin=margin, distance=distance, type_of_triplets="semihard"
    # )
    # cfg.miner = mining_func
    # criterion = losses.TripletMarginLoss(
    #     margin=margin, distance=distance, reducer=reducer
    # )

    ###

    ### OPTIMIZER ###
    classifier = net.get_classifier()

    if str(net) == "ProjectionNet":
        param_groups = [
            {
                "params": net.backbone.parameters(),
                "lr": cfg.backbone_lr,
            },
            {
                "params": list(classifier.parameters()) + list(criterion.parameters()),
                "lr": cfg.head_lr,
            },
        ]
    else:
        if "levit" in cfg.net.backbone_name:
            param_groups = [
                {
                    "params": list(net.parameters())[:-8],
                    "lr": cfg.backbone_lr,
                },
                {
                    "params": list(classifier[0].parameters())
                    + list(classifier[1].parameters()),
                    "lr": cfg.head_lr,
                },
            ]
        else:
            param_groups = [
                {
                    "params": list(net.parameters())[:-2],
                    "lr": cfg.backbone_lr,
                },
                {
                    "params": list(classifier.parameters())
                    + list(criterion.parameters()),
                    "lr": cfg.head_lr,
                },
            ]

    # optimizer = torch.optim.AdamW(param_groups, cfg.head_lr)
    optimizer = Lion(param_groups, cfg.head_lr)
    ###

    ### SCHEDULER ###
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=1e-8
    )
    ###

    cfg.net = net
    cfg.criterion = criterion
    cfg.optimizer = optimizer
    cfg.scheduler = scheduler

    trainer = Trainer(cfg)
    Path(cfg.exp_dir).mkdir(exist_ok=True)
    Path(cfg.exp_dir / "run.py").write_text(Path(__file__).read_text())
    Path(cfg.exp_dir / "dataset.py").write_text(
        (Path(__file__).parent / "src/dataset.py").read_text()
    )
    Path(cfg.exp_dir / "training.py").write_text(
        (Path(__file__).parent / "src/training.py").read_text()
    )
    Path(cfg.exp_dir / "losses.py").write_text(
        (Path(__file__).parent / "src/losses.py").read_text()
    )

    print(f"EXP: {cfg.exp_dir.name}")
    trainer.train()

# epoch 0 - 0.65
# 11,4592 -> 45,8366
