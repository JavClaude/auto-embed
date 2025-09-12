from typing import Literal


class Data:
    type: Literal["csv", "parquet"]
    path: str


class TrainingData(Data):
    pass


class PredictionData(Data):
    pass
