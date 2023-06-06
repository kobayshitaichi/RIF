
from src.utils.config import get_config
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
import torch
import yaml
import warnings

warnings.simplefilter("ignore")
from src.libs.model import ImageTransform, SurgeryDataset, Net
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb
import matplotlib.pyplot as plt

config_path = "./config/config.yaml"
with open(config_path, mode="r") as f:
    d = yaml.load(f, Loader=yaml.FullLoader)
hparams = get_config(config_path)
if hparams.wandb:
    wandb.init(project="RIF", name=hparams.name, config=d)
    wandb_logger = WandbLogger(name=hparams.name, project="RIF")
    logger = [wandb_logger]
else:
    logger = []
train_dataset = SurgeryDataset(
    hparams, transform=ImageTransform(hparams.input_size), phase="train"
)

valid_dataset = SurgeryDataset(
    hparams, transform=ImageTransform(hparams.input_size), phase="valid"
)

test_dataset = SurgeryDataset(
    hparams, transform=ImageTransform(hparams.input_size), phase="test"
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

net = Net(hparams, video_name=hparams.video_name)
early_stop_callback = EarlyStopping(
    monitor=hparams.early_stopping_metric, min_delta=0.00, patience=5, mode="min"
)

trainer = pl.Trainer(
    fast_dev_run=False,
    auto_lr_find=False,
    max_epochs=hparams.max_epocks,
    min_epochs=hparams.min_epocks,
    logger=logger,
    callbacks=[early_stop_callback],
    accelerator="gpu",
    devices=hparams.gpus,
)
# lr_find メソッドを明示的に呼んで学習率探索
lr_finder = trainer.tuner.lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
    method="fit",
)

# グラフの可視化 (要matplotlib)
fig = lr_finder.plot(suggest=True)
plt.savefig("./lr.png")
net.lr = lr_finder.suggestion()

if hparams.train:
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    torch.save(net.state_dict(), hparams.root_dir + "/models/" + hparams.name + ".pth")
    trainer.test(net, dataloaders=test_dataloader)
else:
    net.load_state_dict(torch.load(hparams.root_dir + "/models/" + hparams.name+ ".pth"))
    trainer.test(net, dataloaders=test_dataloader)
