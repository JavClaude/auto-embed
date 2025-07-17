from logging import Logger

from kink import inject
import pandas as pd

from autoembed.src.domain.interfaces.data_repository_interface import (
    DataRepositoryInterface,
)


@inject()
class DataRepositoryLocalCSVAdapter(DataRepositoryInterface):
    def __init__(self, logger: Logger):
        self.logger = logger

    def get_training_data(self, path: str) -> pd.DataFrame:
        self.logger.info(f"Getting training data from {path}")
        return pd.read_csv(path)

    def get_prediction_data(self, path: str) -> pd.DataFrame:
        self.logger.info(f"Getting prediction data from {path}")
        return pd.read_csv(path)
