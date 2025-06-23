from abc import ABC, abstractmethod

from recommendations.src.domain.dataset_preprocessor import DatasetPreprocessor
from recommendations.src.domain.interfaces.classified_embedding_model_interface import (
    ClassifiedEmbeddingModelInterface,
)


class ModelRegistryInterface(ABC):
    @abstractmethod
    def save_preprocessor(self, preprocessor: DatasetPreprocessor, model_id: str) -> None:
        pass

    @abstractmethod
    def load_preprocessor(self, model_id: str | None = None) -> DatasetPreprocessor:
        pass

    @abstractmethod
    def save_model(self, model: ClassifiedEmbeddingModelInterface, model_id: str) -> None:
        pass

    @abstractmethod
    def load_model(self, model: ClassifiedEmbeddingModelInterface, model_id: str | None = None) -> ClassifiedEmbeddingModelInterface:
        pass
