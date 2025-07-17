from abc import ABC, abstractmethod

import pandas as pd


class DataRepositoryInterface(ABC):
    @abstractmethod
    def get_training_data(self, path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_prediction_data(self, path: str) -> pd.DataFrame:
        pass
