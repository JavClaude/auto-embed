import dataclasses
import json
from logging import Logger
import os

from kink import inject
from recommendations.src.domain.dataset_preprocessor import DatasetPreprocessor
from recommendations.src.domain.interfaces.classified_embedding_model_interface import (
    ClassifiedEmbeddingModelInterface,
)
from recommendations.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)
from recommendations.src.domain.entites.columns import (
    CategoricalColumn,
    CategoricalColumns,
    NumericalColumn,
    NumericalColumns,
)


@inject()
class LocalModelRegistryAdapter(ModelRegistryInterface):
    def __init__(self, logger: Logger, path: str = "models"):
        self.path = path
        self.logger = logger

        if not os.path.exists(path):
            self.logger.info(f"Creating directory {path}")
            os.makedirs(path)

    def save_preprocessor(self, preprocessor: DatasetPreprocessor, model_id: str) -> None:
        self.logger.info(f"Saving preprocessor for model {model_id}")

        base_path = f"{self.path}/{model_id}"

        if not os.path.exists(base_path):
            self.logger.info(f"Creating directory {base_path}")
            os.makedirs(base_path)

        preprocessor_data = {
            "numerical_columns": [{column_name: dataclasses.asdict(column)} for column_name, column in preprocessor.numerical_columns.columns.items()],
            "categorical_columns": [{column_name: dataclasses.asdict(column)} for column_name, column in preprocessor.categorical_columns.columns.items()],
        }

        with open(f"{base_path}/preprocessor.json", "w") as f:
            json.dump(preprocessor_data, f)

    def load_preprocessor(self, model_id: str) -> DatasetPreprocessor:
        if model_id == "latest":
            model_id = self._get_latest_model_id()

        self.logger.info(f"Loading exported columns for model {model_id}")

        with open(f"{self.path}/{model_id}/preprocessor.json", "r") as f:
            preprocessor_data = json.load(f)

        numerical_columns = NumericalColumns.from_numerical_columns([NumericalColumn(**list(column.values())[0]) for column in preprocessor_data["numerical_columns"]])
        categorical_columns = CategoricalColumns.from_categorical_columns([CategoricalColumn(**list(column.values())[0]) for column in preprocessor_data["categorical_columns"]])

        return DatasetPreprocessor.from_columns(numerical_columns, categorical_columns)

    def save_model(self, model: ClassifiedEmbeddingModelInterface, model_id: str) -> None:
        self.logger.info(f"Saving model {model_id}")
        model.save(f"{self.path}/{model_id}")

    @inject()
    def load_model(self, model: ClassifiedEmbeddingModelInterface, model_id: str | None = None) -> ClassifiedEmbeddingModelInterface:
        if model_id == "latest":
            model_id = self._get_latest_model_id()

        return model.load(f"{self.path}/{model_id}")

    def _get_latest_model_id(self) -> str:
        model_dirs = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))]
        if not model_dirs:
            raise FileNotFoundError(f"No models found in {self.path}. Please train a model first.")
        return max(model_dirs, key=lambda d: os.path.getctime(os.path.join(self.path, d)))
