import gc
import random
from typing import Union, Optional
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SDDataset(Dataset):
    def __init__(
        self,
        root: Path,
        metadata: Union[pd.DataFrame, str, Path],
        stage: str,
        transform: Optional[transforms.Compose] = None,
    ):
        if isinstance(metadata, (str, Path)):
            metadata = pd.read_csv(metadata)

        if stage != "train":  # if want to train all data
            metadata = metadata[metadata["stage"] == stage]

        self.paths = metadata["path"].values
        if not str(root) in self.paths[0]:
            self.paths = [root / p for p in self.paths]
        self.transform = transform
        self.prompts = metadata["prompt"].values
        self.root = root
        self.stage = stage
        self.clusters = (
            metadata["cluster"].values if "cluster" in metadata.columns else None
        )

        # prepare prompts embeddings before training
        self.prompts = self._get_prompts_embeddings(self.prompts)

    def _get_prompts_embeddings(self, prompts):
        prompt_emb_path = self.root / f"{self.stage}_prompt_embs.pth"
        if prompt_emb_path.exists():
            prompts_embeddings = torch.load(prompt_emb_path, map_location="cpu")
        else:
            st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:1")
            prompts_embeddings = st_model.encode(
                prompts, batch_size=512, show_progress_bar=True, convert_to_tensor=True
            ).cpu()
            torch.save(prompts_embeddings, prompt_emb_path)

            del st_model
            # torch.cuda.empty_cache()
            gc.collect()

        return prompts_embeddings

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.clusters is not None:
            target = self.clusters[idx]
        else:
            target = 1
            if self.stage == "train" and random.random() < 0.1:
                # create negative examples
                idx_new = random.randint(0, len(self.prompts) - 1)
                if idx_new != idx:
                    # cos_sim = torch.nn.functional.cosine_similarity(
                    #     self.prompts[idx].unsqueeze(0),
                    #     self.prompts[idx_new].unsqueeze(0),
                    # )[0].item()
                    # if cos_sim < 0.6:
                    idx = idx_new
                    target = -1

        prompt = self.prompts[idx]

        if self.transform:
            image = self.transform(image)

        return image, prompt, target


class SDCollator:
    def __init__(self):
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts, show_progress_bar=False, convert_to_tensor=True
        )
        return images, prompt_embeddings


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_file, transform=None):
        le = LabelEncoder()
        if isinstance(embeddings_file, str):
            metadata = torch.load(embeddings_file, map_location="cpu")
        else:
            metadata = embeddings_file
        self.embeddings = metadata["embeddings"]
        self.labels = le.fit_transform(metadata["labels"])
        self.classes_ = le.classes_
        self.transform = transform

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]

        if self.transform:
            emb = self.transform(emb)

        item = {"features": emb, "label": self.labels[idx]}

        return item
