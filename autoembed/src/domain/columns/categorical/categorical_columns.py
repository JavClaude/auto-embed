import dataclasses
from typing import Dict, List

import pandas as pd

from autoembed.src.domain.columns.categorical.base_categorical_column import BaseCategoricalColumn


@dataclasses.dataclass
class CategoricalColumns:
    columns: Dict[str, BaseCategoricalColumn]

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, columns: List[str]) -> "CategoricalColumns":
        return cls(columns={column: BaseCategoricalColumn.from_series(dataframe[column]) for column in columns})

    @classmethod
    def from_categorical_columns(cls, categorical_columns: List[BaseCategoricalColumn]) -> "CategoricalColumns":
        return cls(columns={column.name: column for column in categorical_columns})
