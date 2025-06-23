from dataclasses import dataclass


@dataclass
class PredictForModelReleaseCommand:
    model_id: str
    date_to_predict: str
