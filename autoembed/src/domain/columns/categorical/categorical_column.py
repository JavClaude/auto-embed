import dataclasses
from typing import Dict

import pandas as pd
from autoembed.src.domain.columns.base_columns import BaseColumn

UNK_VALUE = "unk"


@dataclasses.dataclass
class CategoricalColumn(BaseColumn):
    name: str
    vocabulary: Dict[str, int]
    value_used_to_fill_na: str = UNK_VALUE
    embedding_dim: int = 128

    @classmethod
    def from_series(cls, series: pd.Series) -> "CategoricalColumn":
        series.fillna(UNK_VALUE, inplace=True)
        series = series.astype(str).apply(lambda x: x.lower().strip())
        vocabulary = {value: index for index, value in enumerate(series.unique())}

        if UNK_VALUE not in vocabulary:
            vocabulary[UNK_VALUE] = len(vocabulary)

        return cls(
            name=series.name,
            vocabulary=vocabulary,
            embedding_dim=cls.infer_embedding_dim(vocabulary),
        )

    @staticmethod
    def infer_embedding_dim(vocabulary: Dict[str, int]) -> int:
        # TODO: Améliorer la logique d'inférence de la dimension d'embedding
        if len(vocabulary) < 30:
            return 32
        else:
            return 128

    def transform(self, series: pd.Series) -> pd.Series:
        series.fillna(UNK_VALUE, inplace=True)
        series = series.astype(str).apply(lambda x: x.lower().strip())
        series = series.map(self.vocabulary, na_action="ignore")
        series = series.fillna(self.vocabulary[UNK_VALUE])
        return series
