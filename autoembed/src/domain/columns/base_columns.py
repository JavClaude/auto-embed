from abc import ABC, abstractmethod

import pandas as pd


class BaseColumn(ABC):
    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        pass
