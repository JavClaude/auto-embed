from logging import Logger

import chromadb
from typing import List
from kink import inject
import tqdm

from autoembed.src.domain.entites.embeddings import BusinessEmbeddings, BatchOfEmbeddings
from autoembed.src.domain.interfaces.embeddings_repository_interface import (
    EmbeddingsRepositoryInterface
)


@inject()
class EmbeddingsChromaDbAdapter(EmbeddingsRepositoryInterface):
    def __init__(self, vector_collection_name: str, logger: Logger):
        self.logger = logger
        self.client = chromadb.PersistentClient(path="chroma_db")
        try:
            self.collection = self.client.get_collection(vector_collection_name)
        except Exception as e:
            self.logger.error(f"⚠️ Error getting collection: {e}")
            self.logger.info(f"🔍 Creating collection {vector_collection_name}")
            self.collection = self.client.create_collection(vector_collection_name)
            self.logger.info(f"✅ Collection {vector_collection_name} created")

        self.max_batch_size = 5460

    def get_embeddings(self, id_column_name: str) -> BusinessEmbeddings:
        self.logger.info(f"Getting embeddings for {id_column_name}")

        return BusinessEmbeddings(
            id_column_name=id_column_name,
            embeddings=self.collection.get(ids=[id_column_name], include=["embeddings"])["embeddings"][0],
        )

    def get_most_similar_embeddings_by_id(self, id_column_name: str, n: int = 6) -> List[BusinessEmbeddings]:
        self.logger.info(f"Getting most similar embeddings for {id_column_name}")
        most_similar_ids = self.collection.query(query_embeddings=[id_column_name], n_results=n)["ids"]
        return [id_column_name for id_column_name in most_similar_ids if id_column_name != id_column_name]

    def update_embeddings(self, embeddings: BusinessEmbeddings) -> None:
        self.logger.info(f"Updating embeddings for {embeddings.id_column_name}")
        self.collection.upsert(
            ids=[embeddings.id_column_name],
            metadatas=[embeddings.get_metadata()],
            embeddings=[embeddings.embeddings],
        )

    def update_batch(self, embeddings_batch: BatchOfEmbeddings) -> None:
        self.logger.info(f"Updating batch of {len(embeddings_batch)} embeddings")


        for i in tqdm.tqdm(range(0, len(embeddings_batch), self.max_batch_size), desc="Updating embeddings ⌛"):
            embeddings_to_upsert = embeddings_batch.embeddings[i : i + self.max_batch_size]

            ids_to_upsert = [embedding.id_column_name for embedding in embeddings_to_upsert]
            embedding_to_upsert = [embedding.embeddings for embedding in embeddings_to_upsert]
            metadata_to_upsert = [embedding.get_metadata() for embedding in embeddings_to_upsert]

            self.collection.upsert(
                ids=ids_to_upsert,
                embeddings=embedding_to_upsert,
                metadatas=metadata_to_upsert,
            )

    def get_embeddings_batch(self, ids: List[str]) -> List[BusinessEmbeddings]:
        self.logger.info(f"Getting batch of {len(ids)} embeddings")
        return [
            BusinessEmbeddings(
                id_column_name=id,
                embeddings=self.collection.get(ids=[id])["embeddings"][0],
            )
            for id in ids
        ]

    def get_all_embeddings(self) -> BatchOfEmbeddings:
        existing_embeddings = self.collection.count()
        batch_retrieval_size = 5000
        
        embeddings_batch = BatchOfEmbeddings()

        for i in tqdm.tqdm(range(0, existing_embeddings, batch_retrieval_size), desc="Getting all embeddings ⌛"):
            try:
                batch = self.collection.get(
                    include=["metadatas", "embeddings"],
                    limit=batch_retrieval_size,
                    offset=i
                )
            except Exception as e:
                self.logger.error(f"Error getting batch of embeddings: {e}")
                continue

            try:
                for position, id in enumerate(batch["ids"]):
                    embedding = BusinessEmbeddings(
                        id_column_name=id,
                        embeddings=batch["embeddings"][position],
                        **batch["metadatas"][position],
                    )

                    embeddings_batch.add_embeddings(embedding)
            
            except Exception as e:
                self.logger.error(f"Error getting embeddings: {e}")
                self.logger.error(f"Batch: {batch['ids'][position]}")
                self.logger.error(f"Batch: {batch['embeddings'][position]}")
                self.logger.error(f"Batch: {batch['metadatas'][position]}")
                continue

        return embeddings_batch
