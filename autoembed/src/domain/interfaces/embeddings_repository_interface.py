from abc import ABC, abstractmethod
from typing import List

from autoembed.src.domain.entites.embeddings import BusinessEmbeddings, BatchOfEmbeddings


class EmbeddingsRepositoryInterface(ABC):
    @abstractmethod
    def get_embeddings(self, id_column_name: str) -> BusinessEmbeddings:
        pass

    @abstractmethod
    def get_most_similar_embeddings_by_id(self, id: str, n: int = 10) -> List[str]:
        pass

    @abstractmethod
    def update_embeddings(self, embeddings: BusinessEmbeddings) -> None:
        pass

    @abstractmethod
    def update_batch(self, embeddings_batch: BatchOfEmbeddings) -> None:
        pass

    @abstractmethod
    def get_embeddings_batch(self, ids: List[str]) -> List[BusinessEmbeddings]:
        pass

    @abstractmethod
    def get_all_embeddings(self) -> BatchOfEmbeddings:
        pass
