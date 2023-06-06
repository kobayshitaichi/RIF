import yaml
from PIL import ImageFile

from src.utils.config import get_config

ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings

import torch
import torch.utils.data as data

warnings.simplefilter("ignore")
import os
import pickle
import shutil
from pathlib import Path

import ffmpeg
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.libs.model import ImageTransform, Net, SurgeryDataset, SemiSupervisedDataset
from src.libs.opt import opt
from src.libs.split_videos import split

config_path = "./config/config_main.yaml"

with open(config_path, mode="r") as f:
    d = yaml.load(f, Loader=yaml.FullLoader)
hparams = get_config(config_path)

# make dirs
root_dir = Path(hparams.root_dir)
video_name = hparams.video_name + ".mp4"
video_path = root_dir / "videos" / hparams.video_name / video_name
result_dir = root_dir / "data" / "test" / hparams.video_name / "results"
split_dir = root_dir / "videos" / hparams.video_name / "split_videos"
img_dir = root_dir / "data" / "test" / hparams.video_name / "images"
log_dir = root_dir / "data" / "test" / hparams.video_name / "logs"
label_dir = root_dir / "data" / "test" / hparams.video_name / "labels"
label_on_dir = label_dir / "on"
label_off_dir = label_dir / "off"
video_img_dir = root_dir / "makevideo" / hparams.video_name / "images"

os.makedirs(result_dir, exist_ok=True)
os.makedirs(split_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(label_on_dir, exist_ok=True)
os.makedirs(label_off_dir, exist_ok=True)
os.makedirs(video_img_dir, exist_ok=True)

# Split videos
if hparams.split:
    split(hparams.video_name, hparams.root_dir)

# Test
if hparams.test:
    test_dataset = SurgeryDataset(
        hparams, transform=ImageTransform(hparams.input_size), phase="test"
    )

    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    net = Net(hparams, video_name=hparams.video_name)
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=hparams.max_epocks,
        min_epochs=hparams.min_epocks,
        accelerator="gpu",
        devices=hparams.gpus,
    )

    net.load_state_dict(torch.load(hparams.root_dir + "/models/train_model.pth"))
    trainer.test(net, dataloaders=test_dataloader)

# Get optical flow
if hparams.opt:
    probe = ffmpeg.probe(video_path)
    duration = probe["format"]["duration"]
    opt_results = []

    print("get optical flow")
    print(duration)
    for n in tqdm(range(int(float(duration) // 5))):
        result = opt(n * 150, (n + 1) * 150, 3, str(video_path))
        tmp = [result] * 150
        opt_results += tmp
        with open(result_dir / "opt_results.pickle", mode="wb") as f:
            pickle.dump(opt_results, f)

    print(len(pd.read_pickle(result_dir / "opt_results.pickle")))


# Semi-supervised learning
if hparams.semi:
    for i in range(hparams.iteration):
        logger = []
        train_dataset = SemiSupervisedDataset(
            hparams, i, transform=ImageTransform(hparams.input_size), phase="train"
        )

        valid_dataset = SemiSupervisedDataset(
            hparams, i, transform=ImageTransform(hparams.input_size), phase="valid"
        )

        test_dataset = SemiSupervisedDataset(
            hparams, i, transform=ImageTransform(hparams.input_size), phase="test"
        )

        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        valid_dataloader = data.DataLoader(
            valid_dataset,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        net = Net(hparams, iteration=i, video_name=hparams.video_name)
        early_stop_callback = EarlyStopping(
            monitor=hparams.early_stopping_metric, min_delta=0.00, patience=3, mode="min"
        )

        trainer = pl.Trainer(
            fast_dev_run=False,
            max_epochs=hparams.max_epocks,
            min_epochs=hparams.min_epocks,
            logger=logger,
            callbacks=[early_stop_callback],
            accelerator="gpu",
            devices=hparams.gpus,
        )


        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
        trainer.test(net, dataloaders=test_dataloader)

# Make video
if hparams.makevideo:
    if not hparams.semi:
        test_dataset = SemiSupervisedDataset(
            hparams, 0, transform=ImageTransform(hparams.input_size), phase="test"
        )
    with open(
        hparams.root_dir
        + "/data/test/"
        + hparams.video_name
        + "/results/preds_bin_{}.pickle".format(str(hparams.iteration - 1)),
        mode="rb",
    ) as f:
        preds_bin = pickle.load(f)[0].tolist()
    path_list = test_dataset.file_list["test"]
    video_path = []
    for i in range(len(path_list)):
        if preds_bin[i] == 1:
            video_path.append(path_list[i])

    for i in range(len(video_path)):
        s = str(i).zfill(6)
        copypath = video_path[i]
        savepath = (
            hparams.root_dir
            + "/makevideo/"
            + hparams.video_name
            + "/images/"
            + s
            + ".jpg"
        )
        shutil.copyfile(copypath, savepath)

    stream = ffmpeg.input(
        hparams.root_dir + "/makevideo/" + hparams.video_name + "/images/%6d.jpg", r=30
    )
    stream = ffmpeg.output(
        stream,
        hparams.root_dir
        + "/makevideo/"
        + hparams.video_name
        + "/"
        + hparams.video_name
        + ".mp4",
    )
    ffmpeg.run(stream)

# Remove dir
if hparams.remove:
    try:
        shutil.rmtree(video_img_dir)
    except:
        pass
    try:
        shutil.rmtree(img_dir)
    except:
        pass
    try:
        shutil.rmtree(label_dir)
    except:
        pass
    try:
        shutil.rmtree(split_dir)
    except:
        pass
