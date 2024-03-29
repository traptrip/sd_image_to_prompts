import logging
from typing import Tuple, Dict, Optional

import torch
import timm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from tqdm.contrib.telegram import tqdm_telegram
import pandas as pd

from src.config import Config
from src import utils
from src.dataset import SDDataset, SDCollator


class Trainer:
    def __init__(self, config: Config) -> None:
        # tqdm
        self.tg_token = config.tg_token
        self.tg_chat_id = config.tg_chat_id

        # dataloader
        self.dataset_dir = config.data_dir
        self.metadata_filepath = config.metadata_filepath

        # transforms
        self.train_transform, self.test_transform = (
            config.transforms["train"],
            config.transforms["test"],
        )

        # exp paths
        self.resume = config.resume
        self.exp_dir = config.exp_dir
        self.pretrained_weights = config.pretrained_weights

        # training params
        self.net = config.net
        self.criterion = config.criterion
        self.optimizer = config.optimizer
        self.clip_value = config.clip_value
        self.scheduler = config.scheduler
        self.miner = config.miner

        self.n_epochs = config.n_epochs
        self.n_warm_epochs = config.n_warm_epochs
        self.n_warm_steps = config.n_warm_steps

        self.device = (
            torch.device(config.device)
            if not isinstance(config.device, torch.device)
            else config.device
        )
        self.batch_size = config.batch_size
        self.n_workers = config.n_workers
        self.use_amp = config.use_amp

        utils.set_seed(config.seed)

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        metadata = pd.read_csv(self.metadata_filepath)

        train_dataset = SDDataset(
            root=self.dataset_dir,
            metadata=metadata,
            stage="train",
            transform=self.train_transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
            drop_last=True,
        )

        eval_dataset = SDDataset(
            root=self.dataset_dir,
            metadata=metadata,
            stage="test",
            transform=self.test_transform,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )
        self.train_steps = len(train_loader)
        self.eval_steps = len(eval_loader)

        if not self.n_warm_steps:
            self.n_warm_steps = self.n_warm_epochs * self.train_steps

        return train_loader, eval_loader

    def _load_exp(
        self,
    ) -> Tuple[int, Dict[str, float]]:
        ckpt_path = self.exp_dir / "last_checkpoint.pth"
        model_path = self.exp_dir / "last_model.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model_ckpt = torch.load(model_path, map_location=self.device)
            metrics = ckpt["metrics"]
            epoch = ckpt["epoch"] + 1
            self.net.load_state_dict(model_ckpt["net_state"])
            self.net.to(self.device)
            self.criterion.load_state_dict(ckpt["criterion_state"])
            self.criterion.to(self.device)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            return epoch, metrics
        else:
            logging.error(f"Failed to load experiment data {self.exp_dir}")
            raise FileNotFoundError(f"No such checkpoint: {ckpt_path}")

    def train_step(
        self,
        epoch: int,
        scaler: Optional[torch.cuda.amp.GradScaler],
        data_loader: DataLoader,
    ):
        self.net.train()

        # train step
        result_loss = 0
        result_score = 0

        pbar = (
            tqdm_telegram(data_loader, token=self.tg_token, chat_id=self.tg_chat_id)
            if self.tg_token
            else tqdm(data_loader)
        )
        for bid, (images, prompts, targets) in enumerate(pbar):
            step = epoch * self.train_steps + bid + 1
            if step <= self.n_warm_steps:
                try:
                    warm_net_params = list(self.net.get_classifier().parameters())
                except:
                    warm_net_params = list(self.net.get_classifier()[0].l.parameters())
                if 0 <= step < self.n_warm_steps:
                    for param in list(
                        set(self.net.parameters()).difference(set(warm_net_params))
                    ):
                        param.requires_grad = False
                elif step == self.n_warm_steps:
                    for param in self.net.parameters():
                        param.requires_grad = True

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                images, prompts = images.to(self.device), prompts.to(self.device)
                targets = targets.to(self.device)
                output = self.net(images)
                loss = self.criterion(output, prompts, targets, self.miner)

                score = torch.nn.functional.cosine_similarity(
                    output.detach(), prompts.detach()
                ).mean()

            self.optimizer.zero_grad()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.clip_value:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.clip_value)
                torch.nn.utils.clip_grad_value_(
                    self.criterion.parameters(), self.clip_value
                )

            if scaler:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            result_loss += loss.item()
            result_score += score.item()

            pbar.set_description(
                f"Epoch: {epoch} Train Loss: {(result_loss / (bid + 1)):.6f}  Train Score: {(result_score / (bid + 1)):.6f}"
            )

        result_loss /= self.train_steps
        result_score /= self.train_steps

        return result_loss, result_score

    def eval_step(
        self,
        epoch: int,
        data_loader: DataLoader,
    ):
        self.net.eval()
        with torch.no_grad():
            result_loss = 0
            result_score = 0
            pbar = (
                tqdm_telegram(data_loader, token=self.tg_token, chat_id=self.tg_chat_id)
                if self.tg_token
                else tqdm(data_loader)
            )
            for bid, (images, prompts, targets) in enumerate(pbar):
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    images, prompts = images.to(self.device), prompts.to(self.device)
                    targets = targets.to(self.device)
                    output = self.net(images)
                    loss = self.criterion(output, prompts, targets, self.miner)
                    score = torch.nn.functional.cosine_similarity(
                        output, prompts
                    ).mean()

                result_loss += loss.item()
                result_score += score.item()

                pbar.set_description(
                    f"Epoch: {epoch} Eval Loss: {(result_loss / (bid + 1)):.6f} Eval Score: {(result_score / (bid + 1)):.6f}"
                )

            result_loss /= self.eval_steps
            result_score /= self.eval_steps

        return result_loss, result_score

    def train(self):
        epoch = 0
        metrics = {"best_loss": float("inf"), "best_score": 0}
        train_loader, eval_loader = self._get_dataloaders()

        if self.resume:
            epoch, metrics = self._load_exp()

        tb_writer = SummaryWriter(self.exp_dir / "tensorboard_logs")

        logging.info(f"Training for {self.n_epochs - epoch} epochs.")

        scaler = (
            torch.cuda.amp.GradScaler()
            if self.use_amp and torch.cuda.is_available()
            else None
        )

        while epoch < self.n_epochs:
            # train
            # train_loss, train_score = 0, 0
            train_loss, train_score = self.train_step(epoch, scaler, train_loader)

            # Logging train
            tb_writer.add_scalar("Train/Loss", train_loss, epoch)
            tb_writer.add_scalar("Train/Score", train_score, epoch)

            # evaluate
            eval_loss, eval_score = self.eval_step(epoch, eval_loader)
            # eval_loss = self.eval_step(epoch, eval_loader)

            # scheduler step
            if self.scheduler is not None:
                if "ReduceLROnPlateau" in str(self.scheduler):
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()

            # Logging valid
            tb_writer.add_scalar("Eval/Loss", eval_loss, epoch)
            tb_writer.add_scalar("Eval/Score", eval_score, epoch)
            tb_writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # Best net save
            # if metrics["best_score"] < eval_score:
            if metrics["best_loss"] > eval_loss:
                metrics["best_loss"] = eval_loss
                metrics["best_score"] = eval_score
                checkpoint = {
                    "epoch": epoch,
                    "metrics": metrics,
                    # "net_state": net.state_dict(),
                    "criterion_state": self.criterion.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None,
                }

                torch.save(
                    {"net_state": self.net.state_dict()},
                    self.exp_dir / "best_model.pth",
                )
                torch.save(checkpoint, self.exp_dir / "best_checkpoint.pth")

            # save last
            checkpoint = {
                "epoch": epoch,
                "metrics": metrics,
                # "net_state": net.state_dict(),
                "criterion_state": self.criterion.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
            }

            torch.save(
                {"net_state": self.net.state_dict()},
                self.exp_dir / "last_model.pth",
            )
            torch.save(checkpoint, self.exp_dir / "last_checkpoint.pth")

            epoch += 1

        tb_writer.close()
        logging.info(f"Training was finished")

    def eval(self):
        metadata = pd.read_csv(self.metadata_filepath)
        eval_dataset = SDDataset(
            root=self.dataset_dir,
            metadata=metadata,
            stage="test",
            transform=self.test_transform,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )
        self.eval_steps = len(eval_loader)
        _ = self._load_exp()
        eval_loss, eval_score = self.eval_step(-1, eval_loader)
        print(f"Eval Loss: {eval_loss:.4f} Eval Score: {eval_score:.4f}")
