import numpy as np
import pandas as pd
import tqdm

from autoembed.src.domain.entities.id_columns import IdColumns
from autoembed.src.domain.entities.metadata_columns import MetadataColumns
from autoembed.src.domain.models.batch_business_embeddings import BatchBusinessEmbeddings
from autoembed.src.domain.models.business_embeddings import BusinessEmbeddings


class BatchBusinessEmbeddingService:
    def generate_batch_business_embeddings(self, id_columns: IdColumns, metadata_columns: MetadataColumns, embeddings: np.ndarray, prediction_data: pd.DataFrame) -> BatchBusinessEmbeddings:
        embeddings_batch = BatchBusinessEmbeddings()

        print(type(id_columns))
        we_need_to_build_the_id_column_from_multiple_columns = len(id_columns.columns) > 1

        embeddings_metadata = prediction_data[id_columns.columns + metadata_columns.columns].to_dict(orient="records")

        for metadata, embedding in tqdm.tqdm(zip(embeddings_metadata, embeddings), desc="Generating embeddings"):
            if we_need_to_build_the_id_column_from_multiple_columns:
                embedding_id = "-".join([str(metadata[id_column]) for id_column in id_columns.columns])
            else:
                embedding_id = metadata.pop(id_columns.columns[0])
            business_embedding = BusinessEmbeddings(
                id=str(embedding_id),
                embeddings=embedding,
                metadata=metadata,
            )
            embeddings_batch.add_embeddings(business_embedding)

        return embeddings_batch
