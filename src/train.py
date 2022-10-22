import pytorch_lightning as pl
from src.datamodule import AminoAcidSequenceDataModule
from src.model import LitProtBert
from pytorch_lightning.loggers.wandb import WandbLogger


def train(pl_model, data_module):
    trainer = pl.Trainer(
        accelerator="gpu",
        # auto_select_gpus=True,
        auto_scale_batch_size="binsearch",
        precision=16,
        auto_select_gpus=True,
        logger=WandbLogger(save_dir="wandb_saves", project="Novozymes Competition"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="saved_models", monitor="validation/loss", mode="min"
            ),
            pl.callbacks.EarlyStopping(
                monitor="validation/loss", mode="min", patience=20
            ),
            pl.callbacks.ModelCheckpoint("saved_models"),  # To save latest model
            pl.callbacks.ModelSummary(),
        ],
        # fast_dev_run=True,
    )
    # trainer.tune(model=pl_model, datamodule=data_module)
    trainer.fit(pl_model, data_module)
    return pl_model


# def execute_task():
model = LitProtBert()
data_module = AminoAcidSequenceDataModule(
    train_dir="data/train.csv", prediction_dir="data/test.csv"
)
train(model, data_module)
