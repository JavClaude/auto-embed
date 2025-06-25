from logging import Logger

from kink import inject
import pandas as pd

from recommendations.src.domain.interfaces.classified_repository_interface import (
    ClassifiedRepositoryInterface,
)


@inject()
class ClassifiedRepositoryLocalAdapter(ClassifiedRepositoryInterface):
    def __init__(self, path: str, logger: Logger):
        self.path = path
        self.logger = logger

    def get_classified_training_data(self, online_date: str) -> pd.DataFrame:
        self.logger.info(f"Getting classified training data from {self.path}")
        return pd.read_csv(f"{self.path}/training/classified_mars_juin_2025.csv")

    def get_classified_prediction_data(self, date_to_predict: str) -> pd.DataFrame:
        self.logger.info(f"Getting classified prediction data from {self.path}")
        return pd.read_csv(f"{self.path}/prediction/classified_online_23_06.csv")
