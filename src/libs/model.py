import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
from pathlib import Path
import shutil

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchmetrics
from PIL import Image
from torchvision import transforms


class ImageTransform(object):
    def __init__(
        self, resize, mean=[0.3164, 0.3875, 0.3762], std=[0.2838, 0.2543, 0.2714]
    ):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    # transforms.ColorJitter(
                    #     brightness=0.5, contrast=0.5, saturation=0.5
                    # ),
                    # transforms.GaussianBlur(kernel_size=3),
                    transforms.Resize((resize, resize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize((resize, resize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize((resize, resize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


class SurgeryDataset(data.Dataset):
    def __init__(self, hparams, transform=None, phase="test"):
        self.phase = phase
        self.file_list = {}
        self.transform = transform
        self.classes = ["on", "off"]
        self.phase = phase
        self.hparams__ = hparams
        self.make_filepath_list()
        self.make_testfilepath_list()

    def make_filepath_list(self):
        root_dir = Path(self.hparams__.root_dir)
        dataset_dir = root_dir / "dataset" / "train" / "images"
        on_dir = dataset_dir / "on"
        off_dir = dataset_dir / "off"
        num_samples = len(os.listdir(off_dir))
        on_img_list = list(on_dir.glob("*.jpg"))[:num_samples]
        off_img_list = list(off_dir.glob("*.jpg"))
        num_split = int(num_samples * 0.8)
        train_file_list = on_img_list[:num_split] + off_img_list[:num_split]
        valid_file_list = on_img_list[num_split:] + off_img_list[num_split:]
        self.file_list["train"] = train_file_list
        self.file_list["valid"] = valid_file_list

    def make_testfilepath_list(self):
        root_dir = Path(self.hparams__.root_dir)
        dataset_dir = root_dir / "dataset" / "test" / "video09" / "images"
        test_file_list = sorted(list(dataset_dir.glob("*.jpg")))
        self.file_list["test"] = test_file_list

    def __len__(self):
        return len(self.file_list[self.phase])

    def __getitem__(self, index):
        img_path = self.file_list[self.phase][index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        if self.phase == "train" or self.phase == "valid":
            label = str(self.file_list[self.phase][index]).split("_")[-2]
        else:
            label = str(self.file_list[self.phase][index]).split("_")[-1][:-4]
        label = self.classes.index(label)

        return img_transformed, label




class SemiSupervisedDataset(data.Dataset):
    def __init__(self, hparams, iteration, transform=None, phase="test"):
        self.phase = phase
        self.file_list = {}
        self.transform = transform
        self.classes = ["on", "off"]
        self.phase = phase
        self.hparams__ = hparams
        self.make_testfilepath_list()
        self.opt_make_train_dataset(iteration)

    def opt_make_train_dataset(self, i):
        with open(
            self.hparams__.root_dir
            + "/dataset/test/"
            + self.hparams__.video_name
            + "/results/opt_results.pickle",
            mode="rb",
        ) as f:
            opt_results = pickle.load(f)

        shutil.rmtree(
            self.hparams__.root_dir
            + "/dataset/test/"
            + self.hparams__.video_name
            + "/labels/on"
        )
        shutil.rmtree(
            self.hparams__.root_dir
            + "/dataset/test/"
            + self.hparams__.video_name
            + "/labels/off"
        )
        os.makedirs(
            self.hparams__.root_dir
            + "/dataset/test/"
            + self.hparams__.video_name
            + "/labels/on",
            exist_ok=True,
        )
        os.makedirs(
            self.hparams__.root_dir
            + "/dataset/test/"
            + self.hparams__.video_name
            + "/labels/off",
            exist_ok=True,
        )

        if i == 0:
            with open(
                self.hparams__.root_dir
                + "/dataset/test/"
                + self.hparams__.video_name
                + "/results/preds_bin.pickle",
                mode="rb",
            ) as f:
                preds_bin = pickle.load(f)[0].tolist()
        else:
            with open(
                self.hparams__.root_dir
                + "/dataset/test/"
                + self.hparams__.video_name
                + "/results/preds_bin_{}.pickle".format(str(i - 1)),
                mode="rb",
            ) as f:
                preds_bin = pickle.load(f)[0].tolist()

        for n in range(len(opt_results)):
            if n % 150 == 0:
                print("make dataset {} / {}".format(n, len(opt_results)))
            if opt_results[n]:
                if preds_bin[n] == 0:
                    shutil.copy(
                        self.file_list["test"][n],
                        self.hparams__.root_dir
                        + "/dataset/test/"
                        + self.hparams__.video_name
                        + "/labels/on",
                    )
            else:
                for j in range(150):
                    if preds_bin[n] == 1:
                        shutil.copy(
                            self.file_list["test"][n],
                            self.hparams__.root_dir
                            + "/dataset/test/"
                            + self.hparams__.video_name
                            + "/labels/off",
                        )

    def make_filepath_list(self):
        root_dir = Path(self.hparams__.root_dir)
        dataset_dir = root_dir / "dataset" / "test" / "labels"
        on_dir = dataset_dir / "on"
        off_dir = dataset_dir / "off"
        num_samples = len(os.listdir(off_dir))
        on_img_list = list(on_dir.glob("*.jpg"))[:num_samples]
        off_img_list = list(off_dir.glob("*.jpg"))
        num_split = int(num_samples * 0.8)
        train_file_list = on_img_list[:num_split] + off_img_list[:num_split]
        valid_file_list = on_img_list[num_split:] + off_img_list[num_split:]
        self.file_list["train"] = train_file_list
        self.file_list["valid"] = valid_file_list

    def make_testfilepath_list(self):
        root_dir = Path(self.hparams__.root_dir)
        dataset_dir = (
            root_dir / "dataset" / "test" / self.hparams__.video_name / "images"
        )
        test_file_list = sorted(list(dataset_dir.glob("*.jpg")))
        self.file_list["test"] = test_file_list

    def __len__(self):
        # 画像の枚数を返す
        return len(self.file_list[self.phase])

    def __getitem__(self, index):
        img_path = self.file_list[self.phase][index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        if self.phase == "train" or self.phase == "valid":
            label = str(self.file_list[self.phase][index]).split("_")[-2]
        else:
            label = str(self.file_list[self.phase][index]).split("_")[-1][:-4]
        label = self.classes.index(label)

        return img_transformed, label


class Net(pl.LightningModule):
    def __init__(self, hparams, iteration=-1, video_name=None):
        super().__init__()
        self.hparams__ = hparams
        self.lr = self.hparams__.learning_rate
        self.iteration = iteration
        self.model = timm.create_model(
            self.hparams__.model_name, pretrained=True, num_classes=2
        )
        self.video_name = str(video_name)
        self.current_preds = []
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.init_metrics()

    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")

    # 順伝搬処理を記述
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hot = F.one_hot(y, 2)
        y_hat = self.forward(x)#.squeeze(1)
        y = y_hot.type_as(y_hat)
        loss = self.bce_loss(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.train_f1(y_hat, y)
        self.log("train_f1", self.train_f1, on_epoch=True, on_step=False)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hot = F.one_hot(y, 2)
        y_hat = self.forward(x).squeeze(1)
        y = y_hot.type_as(y_hat)
        loss = self.bce_loss(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)
        self.val_f1(y_hat, y)
        self.log("val_f1", self.val_f1, on_epoch=True, on_step=False)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = self.forward(x)
            y_hat = self.sigmoid(y_hat)

        y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
        self.current_preds.extend(np.asarray(y_pred).tolist())

    def test_epoch_end(self, outputs):
        save_path = (
            Path(self.hparams__.root_dir) / "dataset" / "test" / self.video_name / "results"
        )
        save_path.mkdir(exist_ok=True)
        if self.iteration >= 0:
            save = 'preds_bin_' + str(self.iteration) + '.pickle'
            save_path_vid = save_path / save
        else:
            save_path_vid = save_path / "preds_bin.pickle"
        save_path_vid = save_path / "preds_bin.pickle"
        print(len(self.current_preds))
        with open(save_path_vid, "wb") as f:
            pickle.dump(
                [
                    np.asarray(self.current_preds),
                ],
                f,
            )
        self.log("test_acc", float(self.test_acc.compute()))
        self.log("test_f1", float(self.test_f1.compute()))

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)

        return optimizer
