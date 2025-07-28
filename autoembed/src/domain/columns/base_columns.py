from abc import ABC, abstractmethod

import pandas as pd


class BaseColumn(ABC):
    @classmethod
    @abstractmethod
    def from_series(cls, series: pd.Series) -> "BaseColumn":
        pass

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        pass
