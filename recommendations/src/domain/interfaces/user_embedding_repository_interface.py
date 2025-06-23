from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class UserEmbedding:
    user_ref: str
    embedding: np.ndarray

class UserEmbeddingRepositoryInterface(ABC):
    @abstractmethod
    def get_user_embedding(self, user_ref: str) -> UserEmbedding:
        pass

    @abstractmethod
    def update_user_embedding(self, user: UserEmbedding) -> None:
        pass

    @abstractmethod
    def get_batch_user_embeddings(self, user_refs: list[str]) -> dict[str, UserEmbedding]:
        pass

    @abstractmethod
    def update_batch_user_embeddings(self, users_embeddings: List[UserEmbedding]) -> None:
        pass
