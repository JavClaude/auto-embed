from abc import ABC, abstractmethod

from autoembed.src.domain.dataset_preprocessor import DatasetPreprocessor
from autoembed.src.domain.interfaces.embedding_model_interface import (
    EmbeddingModelInterface,
)


class ModelRegistryInterface(ABC):
    @abstractmethod
    def save_model_and_preprocessor(self, model: EmbeddingModelInterface, preprocessor: DatasetPreprocessor, model_registry_name: str) -> None:
        pass

    @abstractmethod
    def load_preprocessor(self, model_registry_name: str, model_id: str | None = None) -> DatasetPreprocessor:
        pass

    @abstractmethod
    def load_model(self, model: EmbeddingModelInterface, model_registry_name: str, model_id: str | None = None) -> EmbeddingModelInterface:
        pass
