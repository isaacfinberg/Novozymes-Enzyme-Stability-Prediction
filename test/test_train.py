import pytorch_lightning as pl
from src.datamodule import AminoAcidSequenceDataModule
from src.model import LitProtBert
from transformers import BertTokenizer
import pytest

from src.train import train
import pandas as pd
import torch
from src.dataset import AminoAcidSequenceDataset
from torch.utils.data import DataLoader

data_module = AminoAcidSequenceDataModule(
    train_dir="data/train.csv", prediction_dir="data/test.csv"
)


def test_train():
    model = LitProtBert()
    train(model, data_module)
    assert False


def test_train_process():
    train_data = pd.read_csv("data/train.csv")
    raw_sequences, train_targets = (
        train_data["protein_sequence"].to_list(),
        train_data["tm"].to_list(),
    )

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    train_sequences = [sequence.replace("", " ")[1:-1] for sequence in raw_sequences]
    train_targets = torch.Tensor(train_targets)
    dataset = AminoAcidSequenceDataset(train_sequences, train_targets)
    loader = iter(DataLoader(dataset))
    samples = [next(loader) for _ in range(10)]
    assert len(samples) == 10
    for sample in samples:
        print(sample[0].input_ids.shape)
        # print(sample[0].input_ids.tolist()[0][0])
        # print(tokenizer.decode(sample[0].input_ids.tolist()[0][0]))
    for sample in samples:
        actual = sample[0].input_ids.squeeze(dim=1)
        assert actual.shape == torch.Size([1, 2048]), f"{actual.shape}"


from collections import Counter


@pytest.mark.skip()
def test_sequence_length_expectations():
    train_data = pd.read_csv("data/train.csv")
    raw_sequences, train_targets = (
        train_data["protein_sequence"].to_list(),
        train_data["tm"].to_list(),
    )
    counts = dict()
    higher_than_2048 = 0
    for i, sequence in enumerate(raw_sequences):
        seq_len = len(sequence)
        counts[seq_len] = counts.get(seq_len, 0) + 1
        if seq_len > 2048:
            higher_than_2048 += 1
        if seq_len == 32767:
            highest_idx = i

    print(max(counts.keys()))
    print(highest_idx)
    print(raw_sequences[highest_idx])
    print(higher_than_2048)
    print(higher_than_2048 / len(raw_sequences))
    assert False
