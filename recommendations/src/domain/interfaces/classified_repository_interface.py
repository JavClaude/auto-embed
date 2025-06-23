from abc import ABC, abstractmethod

import pandas as pd


class ClassifiedRepositoryInterface(ABC):
    @abstractmethod
    def get_classified_training_data(self, online_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_classified_prediction_data(self, date_to_predict: str) -> pd.DataFrame:
        pass
