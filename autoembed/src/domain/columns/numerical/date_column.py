import dataclasses

import numpy as np
import pandas as pd


from autoembed.src.domain.columns.base_columns import BaseColumn


NULL_DATE_INSERT_BY_PANDAS = "NaT"


@dataclasses.dataclass
class DateColumn(BaseColumn):
    name: str
    mean_year: float
    mean_month: float
    year_reference: float = 1900
    scaling_year_factor: float = 100
    is_date: bool = True

    @classmethod
    def from_series(cls, series: pd.Series) -> "DateColumn":
        series_convert_to_pandas_timestamp = series.apply(lambda x: pd.to_datetime(x, errors="coerce"))
        mean_month = round(series_convert_to_pandas_timestamp.apply(lambda date: date.month).mean())

        year_series = series_convert_to_pandas_timestamp.apply(lambda date: date.year)
        mean_year = year_series.mean()
        return DateColumn(name=series.name, mean_year=mean_year, mean_month=mean_month)

    def transform(self, series: pd.Series) -> pd.Series:
        series_converted_to_timestamp = series.apply(lambda x: pd.to_datetime(x, errors="coerce"))
        month_series = series_converted_to_timestamp.apply(lambda date: self._encode_cyclical_month(date))
        year_series = series_converted_to_timestamp.apply(lambda date: self._encode_linear_year(date))

        return year_series + month_series

    def _encode_cyclical_month(self, date: pd.Timestamp) -> float:
        if self._is_date_null(date):
            month = self.mean_month
        else:
            month = date.month

        cos_series = np.cos(month * (2 * np.pi / 12))
        sin_series = np.sin(month * (2 * np.pi / 12))

        return cos_series + sin_series

    def _encode_linear_year(self, date: pd.Timestamp) -> int:
        if self._is_date_null(date):
            year = self.mean_year
        else:
            year = date.year

        return (year - self.year_reference) / self.scaling_year_factor

    @staticmethod
    def _is_date_null(date: pd.Timestamp):
        if str(date) == NULL_DATE_INSERT_BY_PANDAS:
            return True
        return False
