from transformers import BertModel
import pytorch_lightning as pl
from typing import OrderedDict, Tuple
import torch


class LitProtBert(pl.LightningModule):
    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = criterion or torch.nn.MSELoss()  # Set default loss to MSE
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.linear_probe = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear1",
                        torch.nn.Linear(in_features=1024, out_features=1024),
                    ),
                    ("dropout1", torch.nn.Dropout()),
                    ("batch_norm1", torch.nn.BatchNorm1d(1024)),
                    # ("selu", torch.nn.SELU()),
                    # ("batch_norm2", torch.nn.BatchNorm1d(1024)),
                    ("linear2", torch.nn.Linear(1024, 1)),
                ]
            )
        )
        self.is_multi_gpu = torch.cuda.device_count() > 1

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_embedding = self.model(
                input_ids=input_ids.squeeze(dim=1),
                attention_mask=attention_mask,
            ).pooler_output  # .flatten(1, 2)
        output = self.linear_probe(bert_embedding)
        return output

    def training_step(
        self: pl.LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        xs, ys = batch  # unpack the batch
        outs = self.common_step(xs)  # apply the model
        loss = self.criterion(outs, ys)
        self.log(
            "train/loss",
            loss,
            # on_step=False,
            on_epoch=True,
            sync_dist=self.is_multi_gpu,
        ),
        return loss

    def validation_step(
        self: pl.LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        xs, ys = batch  # unpack the batch
        outs = self.common_step(xs)  # apply the model
        loss = self.criterion(outs, ys)
        self.log(
            "validation/loss",
            loss,
            # on_step=False,
            # on_epoch=True,
            sync_dist=self.is_multi_gpu,
        ),
        return loss

    def common_step(self, xs):
        return self(
            input_ids=xs.input_ids.squeeze(dim=1),
            attention_mask=xs.attention_mask,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=3e-4
        )  # https://fsdl.me/ol-reliable-img
        return optimizer
