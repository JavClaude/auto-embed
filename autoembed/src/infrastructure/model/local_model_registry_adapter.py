import os
import json
import uuid
import dataclasses
import datetime
from logging import Logger

from kink import inject

from autoembed.src.domain.dataset_preprocessor import DatasetPreprocessor
from autoembed.src.domain.interfaces.embedding_model_interface import (
    EmbeddingModelInterface,
)
from autoembed.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)
from autoembed.src.domain.entites.columns import (
    CategoricalColumn,
    CategoricalColumns,
    NumericalColumn,
    NumericalColumns,
)


@inject()
class LocalModelRegistryAdapter(ModelRegistryInterface):
    def __init__(self, logger: Logger, base_path: str = "models"):
        self.path = base_path
        self.logger = logger

        if not os.path.exists(base_path):
            self.logger.info(f"Creating directory {base_path}")
            os.makedirs(base_path)

    def save_model_and_preprocessor(self, model: EmbeddingModelInterface, preprocessor: DatasetPreprocessor, model_registry_name: str) -> None:
        model_id = self._generate_model_id()
        self.logger.info(f"Saving model {model_id}")

        path = f"{self.path}/{model_registry_name}/{model_id}"

        if not os.path.exists(path):
            self.logger.info(f"Creating directory {path}")
            os.makedirs(path)

        self._save_preprocessor(preprocessor, path)
        self._save_model(model, path)

    def _save_preprocessor(self, preprocessor: DatasetPreprocessor, path: str) -> None:
        preprocessor_data = {
            "numerical_columns": [{column_name: dataclasses.asdict(column)} for column_name, column in preprocessor.numerical_columns.columns.items()],
            "categorical_columns": [{column_name: dataclasses.asdict(column)} for column_name, column in preprocessor.categorical_columns.columns.items()],
            "categorical_features_loss_weights": preprocessor.categorical_features_loss_weights,
        }

        with open(f"{path}/preprocessor.json", "w") as f:
            json.dump(preprocessor_data, f)

    def _save_model(self, model: EmbeddingModelInterface, path: str) -> None:
        self.logger.info(f"Saving model to {path}")
        model.save(path)

    def load_preprocessor(self, model_registry_name: str, model_id: str) -> DatasetPreprocessor:
        if model_id == "latest":
            model_id = self._get_latest_model_id(model_registry_name=model_registry_name)

        self.logger.info(f"Loading exported columns for model {model_id}")

        with open(f"{self.path}/{model_registry_name}/{model_id}/preprocessor.json", "r") as f:
            preprocessor_data = json.load(f)

        numerical_columns = NumericalColumns.from_numerical_columns([NumericalColumn(**list(column.values())[0]) for column in preprocessor_data["numerical_columns"]])
        categorical_columns = CategoricalColumns.from_categorical_columns([CategoricalColumn(**list(column.values())[0]) for column in preprocessor_data["categorical_columns"]])

        return DatasetPreprocessor.from_columns(numerical_columns, categorical_columns)

    @inject()
    def load_model(self, model: EmbeddingModelInterface, model_registry_name: str, model_id: str | None = None) -> EmbeddingModelInterface:
        if model_id == "latest":
            model_id = self._get_latest_model_id(model_registry_name=model_registry_name)

        return model.load(f"{self.path}/{model_registry_name}/{model_id}")

    def _generate_model_id(self) -> str:
        return f"model-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}-{uuid.uuid4()}"

    def _get_latest_model_id(self, model_registry_name: str) -> str:
        model_dirs = [d for d in os.listdir(f"{self.path}/{model_registry_name}") if os.path.isdir(os.path.join(self.path, model_registry_name, d))]
        if not model_dirs:
            raise FileNotFoundError(f"No models found in {self.path}. Please train a model first.")
        return max(model_dirs, key=lambda d: os.path.getctime(os.path.join(self.path, model_registry_name, d)))
