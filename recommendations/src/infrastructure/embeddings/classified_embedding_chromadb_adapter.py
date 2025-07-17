from logging import Logger

import chromadb
from typing import List
from kink import inject
import tqdm

from recommendations.src.domain.interfaces.classified_embeddings_repository_interface import (
    ClassifiedEmbeddings,
    ClassifiedEmbeddingsBatch,
    ClassifiedEmbeddingsRepositoryInterface,
)


@inject()
class ClassifiedEmbeddingsChromaDbAdapter(ClassifiedEmbeddingsRepositoryInterface):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.client = chromadb.PersistentClient(path="chroma_db")
        try:
            self.collection = self.client.get_collection("classified_embeddings")
        except Exception as e:
            self.logger.error(f"Error getting collection: {e}")
            self.logger.info("Creating collection")
            self.collection = self.client.create_collection("classified_embeddings")
            self.logger.info("Collection created")

        self.max_batch_size = 5460

    def get_classified_embeddings(self, classified_ref: str) -> ClassifiedEmbeddings:
        self.logger.info(f"Getting classified embeddings for {classified_ref}")

        return ClassifiedEmbeddings(
            classified_ref=classified_ref,
            classified_embeddings=self.collection.get(ids=[classified_ref], include=["embeddings"])["embeddings"][0],
        )

    def get_most_similar_classified_embeddings(self, classified_embeddings: ClassifiedEmbeddings, n: int = 6) -> List[ClassifiedEmbeddings]:
        self.logger.info(f"Getting most similar classified embeddings for {classified_embeddings.classified_ref}")
        most_similar_classified_refs = self.collection.query(query_embeddings=[classified_embeddings.classified_embeddings], n_results=n)["ids"]
        return [classified_ref for classified_ref in most_similar_classified_refs if classified_ref != classified_embeddings.classified_ref]

    def update_classified_embeddings(self, classified_embeddings: ClassifiedEmbeddings) -> None:
        self.logger.info(f"Updating classified embeddings for {classified_embeddings.classified_ref}")
        self.collection.upsert(
            ids=[classified_embeddings.classified_ref],
            metadatas=[classified_embeddings.get_classified_metadata()],
            embeddings=[classified_embeddings.classified_embeddings],
        )

    def update_batch(self, classified_embeddings_batch: ClassifiedEmbeddingsBatch) -> None:
        self.logger.info(f"Updating batch of {len(classified_embeddings_batch)} classified embeddings")


        for i in tqdm.tqdm(range(0, len(classified_embeddings_batch), self.max_batch_size), desc="Updating classified embeddings"):
            classified_embeddings_to_upsert = classified_embeddings_batch.classified_embeddings[i : i + self.max_batch_size]

            ids_to_upsert = [classified_embedding.classified_ref for classified_embedding in classified_embeddings_to_upsert]
            embedding_to_upsert = [classified_embedding.classified_embeddings for classified_embedding in classified_embeddings_to_upsert]
            metadata_to_upsert = [classified_embedding.get_classified_metadata() for classified_embedding in classified_embeddings_to_upsert]

            self.collection.upsert(
                ids=ids_to_upsert,
                embeddings=embedding_to_upsert,
                metadatas=metadata_to_upsert,
            )

    def get_classified_embeddings_batch(self, classified_refs: List[str]) -> List[ClassifiedEmbeddings]:
        self.logger.info(f"Getting batch of {len(classified_refs)} classified embeddings")
        return [
            ClassifiedEmbeddings(
                classified_ref=classified_ref,
                classified_embeddings=self.collection.get(ids=[classified_ref])["embeddings"][0],
            )
            for classified_ref in classified_refs
        ]

    def get_all_embeddings(self) -> ClassifiedEmbeddingsBatch:
        self.logger.info("Getting all embeddings")

        existing_embeddings = self.collection.count()
        batch_retrieval_size = 5000
        
        classified_embeddings_batch = ClassifiedEmbeddingsBatch()

        for i in tqdm.tqdm(range(0, existing_embeddings, batch_retrieval_size), desc="Getting all embeddings"):
            batch = self.collection.get(
                include=["metadatas", "embeddings"],
                limit=batch_retrieval_size,
                offset=i
            )
            try:
                for position, id in enumerate(batch["ids"]):
                    classified_embedding = ClassifiedEmbeddings(
                        classified_ref=id,
                        classified_embeddings=batch["embeddings"][position],
                        **batch["metadatas"][position],
                    )

                    classified_embeddings_batch.add_classified_embeddings(classified_embedding)
            
            except Exception as e:
                self.logger.error(f"Error getting embeddings: {e}")
                self.logger.error(f"Batch: {batch['ids'][position]}")
                self.logger.error(f"Batch: {batch['embeddings'][position]}")
                self.logger.error(f"Batch: {batch['metadatas'][position]}")
                continue

        return classified_embeddings_batch
