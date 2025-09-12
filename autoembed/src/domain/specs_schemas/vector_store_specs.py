from dataclasses import dataclass
import enum
from typing import List

from autoembed.src.domain.entities.id_columns import IdColumns
from autoembed.src.domain.entities.metadata_columns import MetadataColumns


class SupportedVectorStoreBackend(enum.Enum):
    CHROMA_DB = "chromadb"


@dataclass
class VectorStoreSpecs:
    vector_store_backend: SupportedVectorStoreBackend
    vector_collection_name: str
    id_columns: IdColumns
    metadata_columns: MetadataColumns

    @classmethod
    def from_specs(self, vector_store_backend: str, vector_collection_name: str, id_columns: List[str], metadata_columns: List[str] | None):
        if vector_store_backend not in SupportedVectorStoreBackend:
            raise ValueError(f"Vector backend store: {vector_store_backend} not supported, supported backend are: {[backend.value for backend in SupportedVectorStoreBackend]}")

        if vector_collection_name == "":
            raise ValueError("Vectore store collection should not be emptyyyy")

        one_and_empty_id_column = len(id_columns) == 1 and id_columns[0] == ""

        if one_and_empty_id_column:
            raise ValueError("If only id column is specified, this one cannot be empty small hacker!")

        return VectorStoreSpecs(SupportedVectorStoreBackend(vector_store_backend), vector_collection_name, IdColumns(id_columns), MetadataColumns(metadata_columns))
