import pandas as pd
import torch
from transformers import BertTokenizer


class AminoAcidSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer = 'Rostlab/prot_bert', transform=None, target_transform=None):
        super().__init__()
        raw_sequences = df['protein_sequence'].to_list()
        if 'tm' in df.columns:
            target=df['tm'].to_list()
        else:
            target = None

        self.sequences = [
            sequence.replace("", " ")[1:-1] for sequence in raw_sequences
        ]
        self.target = target
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer, do_lower_case=False
        )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # # sequences = collate_tensors(self.sequences[idx])
        # assert type(sequences) == str
        tokens = self.tokenizer.encode_plus(
            self.sequences[idx],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        # tokens.input_ids = tokens.input_ids.squeeze(dim=1)
        # if self.target is not None:
        # targets = {"target": torch.Tensor(self.target[idx])}
        # targets = torch.Tensor(self.target[idx])
        targets = self.target[idx]
        return (tokens, targets)
        # else:
        #     return tokens

class BertEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, target=None, transform=None, target_transform=None):
        super().__init__()
        self.sequences = sequences
        self.target = target
        