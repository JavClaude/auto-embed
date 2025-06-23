from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class ClassifiedEmbeddings:
    classified_ref: str
    classified_embeddings: np.ndarray
    # TODO: add zip_code


@dataclass
class ClassifiedEmbeddingsBatch:
    classified_refs: List[str]
    classified_embeddings: np.ndarray


class ClassifiedEmbeddingsRepositoryInterface(ABC):
    @abstractmethod
    def get_classified_embeddings(self, classified_ref: str) -> ClassifiedEmbeddings:
        pass

    @abstractmethod
    def get_most_similar_classified_embeddings(self, classified_ref: str, n: int = 10) -> List[ClassifiedEmbeddings]:
        pass

    @abstractmethod
    def update_classified_embeddings(self, classified_embeddings: ClassifiedEmbeddings) -> None:
        pass

    @abstractmethod
    def update_batch(self, classified_embeddings_batch: ClassifiedEmbeddingsBatch) -> None:
        pass

    @abstractmethod
    def get_classified_embeddings_batch(self, classified_refs: List[str]) -> List[ClassifiedEmbeddings]:
        pass
