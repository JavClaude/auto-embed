from abc import ABC, abstractmethod
from typing import List
import numpy as np

from autoembed.src.domain.models.embeddings.batch_business_embeddings import BatchBusinessEmbeddings
from autoembed.src.domain.models.embeddings.business_embeddings import BusinessEmbeddings


class EmbeddingsRepositoryInterface(ABC):
    @abstractmethod
    def get_embeddings(self, id_column_name: str) -> BusinessEmbeddings:
        pass

    @abstractmethod
    def get_most_similar_embeddings(self, embeddings: np.ndarray, n: int = 10) -> List[str]:
        pass

    @abstractmethod
    def get_embeddings_batch(self, ids: List[str]) -> List[BusinessEmbeddings]:
        pass

    @abstractmethod
    def get_all_embeddings(self) -> BatchBusinessEmbeddings:
        pass

    @abstractmethod
    def update_embeddings(self, embeddings: BusinessEmbeddings) -> None:
        pass

    @abstractmethod
    def update_batch(self, embeddings_batch: BatchBusinessEmbeddings) -> None:
        pass
