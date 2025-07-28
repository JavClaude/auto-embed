from abc import ABC, abstractmethod
from typing import List

from autoembed.src.domain.models.batch_business_embeddings import BatchBusinessEmbeddings
from autoembed.src.domain.models.business_embeddings import BusinessEmbeddings


class EmbeddingsRepositoryInterface(ABC):
    @abstractmethod
    def get_embeddings(self, id_column_name: str) -> BusinessEmbeddings:
        pass

    @abstractmethod
    def get_most_similar_embeddings_by_id(self, id: str, n: int = 10) -> List[str]:
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
