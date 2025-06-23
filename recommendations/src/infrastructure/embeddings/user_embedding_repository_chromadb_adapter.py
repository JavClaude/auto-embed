from logging import Logger
from typing import List

import chromadb
from kink import inject

from recommendations.src.domain.interfaces.user_embedding_repository_interface import UserEmbedding, UserEmbeddingRepositoryInterface


@inject()
class UserEmbeddingRepositoryChromaDbAdapter(UserEmbeddingRepositoryInterface):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.client = chromadb.PersistentClient(path="chroma_db")
        try:
            self.collection = self.client.get_collection("user_embeddings")
        except Exception as e:
            self.logger.error(f"Error getting collection: {e}")
            self.logger.info("Creating collection")
            self.collection = self.client.create_collection("user_embeddings")
            self.logger.info("Collection created")

    def get_user_embedding(self, user_ref: str) -> list[float]:
        return self.collection.get(ids=[user_ref], include=["embeddings"])["embeddings"][0]

    def update_user_embedding(self, user: UserEmbedding) -> None:
        self.collection.upsert(ids=[user.user_ref], embeddings=[user.embedding])

    def get_batch_user_embeddings(self, user_refs: list[str]) -> List[UserEmbedding]:
        embeddings = self.collection.get(ids=user_refs, include=["embeddings"])["embeddings"]
        return [UserEmbedding(user_ref=user_ref, embedding=embedding) for user_ref, embedding in zip(user_refs, embeddings)]

    def update_batch_user_embeddings(self, users_embeddings: List[UserEmbedding]) -> None:
        return self.collection.upsert(
            ids=[user.user_ref for user in users_embeddings],
            embeddings=[user.embedding for user in users_embeddings],
        )
