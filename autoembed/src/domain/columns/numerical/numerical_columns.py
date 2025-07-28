import dataclasses
from typing import Dict, List

import pandas as pd

from autoembed.src.domain.columns.numerical.numerical_column import NumericalColumn


@dataclasses.dataclass
class NumericalColumns:
    columns: Dict[str, NumericalColumn]
    numerical_dimensions: int

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, columns: List[str]) -> "NumericalColumns":
        return cls(
            columns={column: NumericalColumn.from_series(dataframe[column]) for column in columns},
            numerical_dimensions=len(columns),
        )

    @classmethod
    def from_numerical_columns(cls, numerical_columns: List[NumericalColumn]) -> "NumericalColumns":
        return cls(
            columns={column.name: column for column in numerical_columns},
            numerical_dimensions=len(numerical_columns),
        )
