from abc import ABC, abstractmethod
from typing import Any

from typing import Dict
from autoembed.src.domain.interfaces.embedding_model_interface import (
    EmbeddingModelInterface,
)


class ModelRegistryInterface(ABC):
    @abstractmethod
    def save_model_and_preprocessor(self, model: EmbeddingModelInterface, preprocessor: Dict[str, Any], model_registry_name: str) -> None:
        pass

    @abstractmethod
    def load_json_preprocessor(self, model_registry_name: str, model_id: str | None = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_model(self, model: EmbeddingModelInterface, model_registry_name: str, model_id: str | None = None) -> EmbeddingModelInterface:
        pass
