import dataclasses
from typing import Dict, List

import pandas as pd

from autoembed.src.domain.columns.numerical.date_column import DateColumn
from autoembed.src.domain.columns.numerical.numerical_column import NumericalColumn


@dataclasses.dataclass
class NumericalColumns:
    columns: Dict[str, NumericalColumn | DateColumn]
    numerical_dimensions: int

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, numerical_columns_names: List[str], date_columns_names: List[str] | None) -> "NumericalColumns":
        columns = {}

        for numerical_column_name in numerical_columns_names:
            columns[numerical_column_name] = NumericalColumn.from_series(dataframe[numerical_column_name])

        if date_columns_names:
            for date_column_name in date_columns_names:
                columns[date_column_name] = DateColumn.from_series(dataframe[date_column_name])

            total_numerical_dimensions = len(numerical_columns_names) + len(date_columns_names)
        else:
            total_numerical_dimensions = len(numerical_columns_names)

        return cls(columns=columns, numerical_dimensions=total_numerical_dimensions)

    @classmethod
    def from_columns(cls, numerical_or_date_columns: List[NumericalColumn | DateColumn]) -> "NumericalColumns":
        columns = {}

        for numerical_column in numerical_or_date_columns:
            columns[numerical_column.name] = numerical_column

        total_numerical_dimensions = len(numerical_or_date_columns)

        return cls(
            columns={column.name: column for column in numerical_or_date_columns},
            numerical_dimensions=total_numerical_dimensions,
        )

    def get_all_columns_names(self) -> List[str]:
        return list(self.columns.keys())
