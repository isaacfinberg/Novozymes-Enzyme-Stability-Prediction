import pandas as pd
import pytest
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import torch
from src.dataset import AminoAcidSequenceDataset
import pickle
import numpy as np

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


def create_embeddings(
    input_csv="data/train.csv",
    pretrained_tokenizer="Rostlab/prot_bert",
    pretrained_model="Rostlab/prot_bert",
):
    # tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer, do_lower_case=False)
    df = pd.read_csv(input_csv)
    model = BertModel.from_pretrained(pretrained_model)
    model.to(device)
    dataset = AminoAcidSequenceDataset(df)
    loader = DataLoader(dataset, batch_size=100, num_workers=12, pin_memory=True)
    bert_embeddings = []
    for batch in loader:
        xs, ys = batch
        xs.to(device)
        with torch.no_grad():
            embeddings = (
                model(
                    input_ids=xs.input_ids.squeeze(dim=1),
                    attention_mask=xs.attention_mask,
                )
                # .pooler_output.detach()
                # .to("cpu")
            )
        embeddings = embeddings.pooler_output.tolist()
        bert_embeddings.extend(embeddings)
        # break
        # assert False, f"{bert_embeddings}"
    df["bert_embedding"] = bert_embeddings
    with open("data/pd_pickles/embeddings.pickle", "wb") as handle:
        pickle.dump(df, handle)
    df.to_csv("data/pd_hdf5s/embedding_csv")
    # df.to_hdf("data/pd_hdf5s/test_hdf5", key="df")
