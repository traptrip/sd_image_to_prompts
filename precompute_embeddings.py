from pathlib import Path

import pandas as pd
from tqdm import tqdm
import open_clip
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from src.dataset import ImagesDataset
from src.config import Config

cfg = Config()
BATCH_SIZE = 256
N_WORKERS = 4


if __name__ == "__main__":
    images_names = [p.name for p in (cfg.data_dir / "images").iterdir()]

    metadata = pd.read_parquet(cfg.metadata_filepath, engine="pyarrow")
    print("Base metadata size:", metadata.shape[0])
    metadata = metadata[metadata.image_name.isin(images_names)]
    metadata["path"] = str(cfg.data_dir / "images") + "/" + metadata["image_name"]
    print("Result metadata size:", metadata.shape[0])

    le = LabelEncoder()
    metadata["label"] = le.fit_transform(metadata["prompt"])
    print(f"Numper of labels(unique prompts): {metadata.label.nunique()}")

    if (
        Path("data/full.csv").exists()
        and Path("data/train.csv").exists()
        and Path("data/test.csv").exists()
    ):
        metadata = pd.read_csv("data/full.csv")

    # initialize embedder model
    embedder = open_clip.create_model(
        cfg.embedder_name, pretrained=cfg.embedder_checkpoint
    )
    embedder = embedder.to(cfg.device)
    embedder = embedder.eval()

    # initialize image transforms
    transform = open_clip.image_transform(
        embedder.visual.image_size,
        is_train=False,
        mean=getattr(embedder.visual, "image_mean", None),
        std=getattr(embedder.visual, "image_std", None),
    )

    # start computing
    images_dataset = ImagesDataset(metadata, transform)
    data_loader = DataLoader(
        images_dataset,
        BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=N_WORKERS,
    )

    embeddings = torch.zeros((len(images_dataset), 768))
    labels = torch.zeros(len(images_dataset))
    for i, batch in enumerate(tqdm(data_loader, desc="Compute embeddings")):
        image = batch["features"].to(cfg.device)
        with torch.no_grad():
            emb = embedder.encode_image(image, normalize=True).cpu()
        embeddings[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = emb
        labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = batch["label"]

        if i % 1000 == 0 or i == 0:
            torch.save(
                {"embeddings": embeddings, "labels": labels},
                cfg.data_dir / "precomputed_embeddings.pth",
            )

    torch.save(
        {"embeddings": embeddings, "labels": labels},
        cfg.data_dir / "precomputed_embeddings.pth",
    )
