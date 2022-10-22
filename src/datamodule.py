import pandas as pd
import pytorch_lightning as pl
import torch

from src.dataset import AminoAcidSequenceDataset
import math


class AminoAcidSequenceDataModule(pl.LightningDataModule):
    def __init__(
        self, train_dir: str, prediction_dir: str, batch_size: int = 99, train_frac=0.8
    ):
        super().__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, return_tensors='pt')
        self.train_dir = train_dir
        self.prediction_dir = prediction_dir
        self.batch_size = batch_size
        self.train_frac = train_frac

    def prepare_data(self):
        train_data = pd.read_csv(self.train_dir)

        # train_targets = torch.Tensor(train_targets).reshape(len(train_targets), 1)
        # xs = re.sub(r"[UZOB]", "X", xs)

        pred_data = pd.read_csv(self.prediction_dir)
        # raw_test_sequences = pred_data["protein_sequence"].to_list()
        # test_sequences = [
        #     sequence.replace("", " ")[1:-1] for sequence in raw_test_sequences
        # ]

        # self.dataset = AminoAcidSequenceDataset(
        #     sequences=train_sequences, target=train_targets
        # )
        # self.prediction_dataset = AminoAcidSequenceDataset(test_sequences)
        self.val_frac = 1 - self.train_frac
        # TODO: FIX THIS, IT BREAKS THE DATALOADERS
        # OR THE LENGTH OF THE INPUT IDS IS DIFFERENT PER INPUT?
        self.train_indices = list(
            range(math.floor(len(self.dataset) * self.train_frac))
        )
        self.val_indices = list(range(self.train_indices[-1], len(self.dataset)))

    def setup(self, stage=None):
        if stage == "fit" or stage is None:  # other stages: "test", "predict"
            self.train_dataset = torch.utils.data.Subset(
                self.dataset, self.train_indices
            )
            self.val_dataset = torch.utils.data.Subset(self.dataset, self.val_indices)

    def train_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=12,
            pin_memory=True,
        )

    def val_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=12,
            pin_memory=True,
        )

    # def predict_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
    #     return torch.utils.data.DataLoader(self.prediction_dataset, batch_size=2*self.batch_size)
