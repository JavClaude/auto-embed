import numpy as np
import pandas as pd
import tqdm

from autoembed.src.domain.models.batch_business_embeddings import BatchBusinessEmbeddings
from autoembed.src.domain.models.business_embeddings import BusinessEmbeddings
from autoembed.src.yaml.auto_embed_yaml_schema import IdColumns, MetadataColumns


class BatchBusinessEmbeddingService:
    def generate_batch_business_embeddings(self, id_columns: IdColumns, metadata_columns: MetadataColumns, embeddings: np.ndarray, prediction_data: pd.DataFrame) -> BatchBusinessEmbeddings:
        embeddings_batch = BatchBusinessEmbeddings()

        we_need_to_build_the_id_column_from_multiple_columns = len(id_columns.columns) > 1

        if we_need_to_build_the_id_column_from_multiple_columns:
            embeddings_metadata = prediction_data[id_columns.columns + metadata_columns].to_dict(orient="records")
        else:
            embeddings_metadata = prediction_data[id_columns.columns[0] + metadata_columns.metadata_columns.columns].to_dict(orient="records")

        for metadata, embedding in tqdm.tqdm(zip(embeddings_metadata, embeddings), desc="Generating embeddings"):
            if we_need_to_build_the_id_column_from_multiple_columns:
                embedding_id = "-".join([str(embeddings_metadata[id_column]) for id_column in id_columns.columns])
            else:
                embedding_id = embeddings_metadata.pop(id_columns.columns[0])

            business_embedding = BusinessEmbeddings(
                id=embedding_id,
                embeddings=embedding,
                metadata=metadata,
            )
            embeddings_batch.add_embeddings(business_embedding)

        return embeddings_batch
