import dataclasses

import pandas as pd

from autoembed.src.domain.columns.base_columns import BaseColumn


@dataclasses.dataclass
class NumericalColumn(BaseColumn):
    name: str
    value_used_to_fill_na: float | int
    mean: float
    std: float

    @classmethod
    def from_series(cls, series: pd.Series) -> "NumericalColumn":
        mean = series.mean()
        series.fillna(mean, inplace=True)
        median = series.median()
        std = series.std()
        return NumericalColumn(name=series.name, value_used_to_fill_na=median, mean=mean, std=std)

    def transform(self, series: pd.Series) -> pd.Series:
        series.fillna(self.value_used_to_fill_na, inplace=True)
        return (series - self.mean) / self.std
