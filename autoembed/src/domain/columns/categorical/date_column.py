import dataclasses

import numpy as np
import pandas as pd

from autoembed.src.domain.columns.base_columns import BaseColumn


@dataclasses.dataclass
class DateColumn(BaseColumn):
    name: str
    default_date_to_use: str = "1900-01-01"

    def _encode_cyclical(self, series: pd.Series, period: int) -> pd.Series:
        series = series.apply(lambda x: np.cos(x * (2 * np.pi / period)))
        series = series.apply(lambda x: np.sin(x * (2 * np.pi / period)))
        return series

    def _encode_linear_year(self, series: pd.Series) -> pd.Series:
        series = series.apply(lambda x: x - 1900)
        return series

    def transform(self, series: pd.Series) -> pd.Series:
        series = series.apply(lambda x: x.strftime("%Y-%m-%d"))
        series = series.apply(lambda x: x if x != "NaT" else self.default_date_to_use)
        series = series.apply(lambda x: pd.to_datetime(x))
        series = self._encode_linear_year(series)
        series = self._encode_cyclical(series, 12)
        series = self._encode_cyclical(series, 100)
        return series
