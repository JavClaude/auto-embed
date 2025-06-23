import fire

from recommendations.src.usescases.commands.prediction.predict_for_model_release_command import (
    PredictForModelReleaseCommand,
)
from recommendations.src.usescases.commands.prediction.predict_for_model_release_usecase import (
    PredictForModelReleaseUsecase,
)


def predict_for_model_release(model_id: str = "latest", date_to_predict: str = "2025-06-19"):
    predict_for_model_release_usecase = PredictForModelReleaseUsecase()
    predict_for_model_release_command = PredictForModelReleaseCommand(model_id, date_to_predict)
    predict_for_model_release_usecase.execute(predict_for_model_release_command)


def main():
    fire.Fire(predict_for_model_release)
