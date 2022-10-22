import pytest
import pandas as pd

from src.create_bert_embeddings import create_embeddings


def test_create_embeddings():
    expected_embedding_len = 2048
    expected_type = pd.DataFrame
    actual = create_embeddings()
