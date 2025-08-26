import dataclasses
from typing import Dict, List

import pandas as pd

from autoembed.src.domain.columns.categorical.categorical_column import CategoricalColumn


@dataclasses.dataclass
class CategoricalColumns:
    columns: Dict[str, CategoricalColumn]

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, columns: List[str]) -> "CategoricalColumns":
        return cls(columns={column: CategoricalColumn.from_series(dataframe[column]) for column in columns})

    @classmethod
    def from_categorical_columns(cls, categorical_columns: List[CategoricalColumn]) -> "CategoricalColumns":
        return cls(columns={column.name: column for column in categorical_columns})

    def get_all_columns_names(self) -> List[str]:
        return list(self.columns.keys())
