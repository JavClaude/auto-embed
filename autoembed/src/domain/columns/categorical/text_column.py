import re
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

from autoembed.src.domain.columns.categorical.base_categorical_column import BaseCategoricalColumn

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
SPECIAL_TOKENS = [CLS_TOKEN, SEP_TOKEN, "[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"]


@dataclass
class TransformedTextColumn:
    input_ids: np.ndarray


@dataclass
class TextColumn(BaseCategoricalColumn):
    name: str
    tokenizer: Tokenizer
    vocab_size: int
    max_length: int = 64
    max_vocab_size: int = 300
    word_embedding: int = 64

    @classmethod
    def from_series(cls, series: pd.Series, max_length: int = 64, vocab_size: int = 10000, word_embedding: int = 64) -> "TextColumn":
        print("TOTOTOOTOTTO")
        print(type(series))
        clean_series = cls._normalize_text_series(series)

        tokenizer = Tokenizer(BPE())

        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2, continuing_subword_prefix="##")

        tokenizer.train_from_iterator(clean_series.tolist(), trainer)

        tokenizer.post_processor = TemplateProcessing(
            single=f"{CLS_TOKEN} $A {SEP_TOKEN}",
            special_tokens=[
                (CLS_TOKEN, tokenizer.token_to_id(CLS_TOKEN)),
                (SEP_TOKEN, tokenizer.token_to_id(SEP_TOKEN)),
            ],
        )

        tokenizer.enable_padding(length=max_length, pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        tokenizer.enable_truncation(max_length=max_length)

        return cls(
            name=series.name,
            tokenizer=tokenizer,
            vocab_size=tokenizer.get_vocab_size(),
            max_length=max_length,
            max_vocab_size=vocab_size,
            word_embedding=word_embedding,
        )

    @staticmethod
    def _normalize_text_series(series: pd.Series) -> pd.Series:
        clean_series = series.fillna("")
        clean_series = clean_series.astype(str)
        clean_series = clean_series.map(TextColumn._normalize_text)
        clean_series = clean_series.replace("", "[EMPTY]")
        return clean_series

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip()
        text = text.lower()
        text = re.sub("\\s+", " ", text)
        return text

    def transform(self, series: pd.Series) -> TransformedTextColumn:
        clean_series = self._normalize_text_series(series)

        encoded = self.tokenizer.encode_batch(clean_series.tolist())

        input_ids = np.array([enc.ids for enc in encoded])
        # attention_mask supprimé - mask_zero=True dans l'Embedding s'en charge

        return TransformedTextColumn(input_ids=input_ids)

    def encode_single_text(self, text: str) -> TransformedTextColumn:
        cleaned_text = self._normalize_text(text)
        encoded = self.tokenizer.encode(cleaned_text)

        return TransformedTextColumn(input_ids=np.array([encoded.ids]))

    def decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_special_token_ids(self) -> Dict[str, int]:
        return {token: self.tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}

    @classmethod
    def from_tokenizer(cls, tokenizer: Tokenizer, name: str, max_length: int = 512, embedding_dim: int = 128) -> "TextColumn":
        return cls(name=name, tokenizer=tokenizer, max_length=max_length, embedding_dim=embedding_dim)
