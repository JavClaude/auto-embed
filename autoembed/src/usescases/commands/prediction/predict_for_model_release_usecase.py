from logging import Logger

import tqdm
from kink import inject

from autoembed.src.domain.entites.embeddings import BatchOfEmbeddings, BusinessEmbeddings
from autoembed.src.domain.interfaces.embedding_model_interface import (
    EmbeddingModelInterface,
)
from autoembed.src.domain.interfaces.embeddings_repository_interface import (
    EmbeddingsRepositoryInterface,
)
from autoembed.src.domain.interfaces.data_repository_interface import (
    DataRepositoryInterface,
)
from autoembed.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)
from autoembed.src.usescases.commands.prediction.predict_for_model_release_command import (
    PredictForModelReleaseCommand,
)


@inject()
class PredictForModelReleaseUsecase:
    def __init__(
        self,
        data_repository: DataRepositoryInterface,
        model_registry: ModelRegistryInterface,
        embedding_model: EmbeddingModelInterface,
        logger: Logger,
        embeddings_repository: EmbeddingsRepositoryInterface,
    ):
        self.logger = logger
        self.data_repository = data_repository
        self.model_registry = model_registry
        self.embeddings_repository = embeddings_repository
        self.embedding_model = embedding_model

    def execute(self, command: PredictForModelReleaseCommand) -> None:
        self.logger.info(f"Predicting for model release {command.project_name} {command.model_version} for {command.prediction_data.path}")

        prediction_data = self.data_repository.get_prediction_data(command.prediction_data.path)
        dataset_preprocessor = self.model_registry.load_preprocessor(command.project_name, command.model_version)
        model = self.model_registry.load_model(self.embedding_model, command.project_name, command.model_version)

        preprocessed_data = dataset_preprocessor.preprocess(prediction_data)
        embeddings = model.embed(preprocessed_data)

        if len(command.id_column.columns) > 1:
            essential_data = prediction_data[command.id_column.columns + command.vector_store.metadata_columns.columns].to_dict(orient="records")
        else:
            essential_data = prediction_data[command.id_column.columns[0] + command.vector_store.metadata_columns.columns].to_dict(orient="records")

        embeddings_batch = BatchOfEmbeddings()

        for essential_data, embedding in tqdm.tqdm(zip(essential_data, embeddings), desc="Generating embeddings"):
            
            the_id_column_needs_to_be_built_from_multiple_columns = len(command.id_column.columns) > 1
            if the_id_column_needs_to_be_built_from_multiple_columns:
                id = "-".join([str(essential_data[column]) for column in command.id_column.columns])
            else:
                id = essential_data[command.id_column.columns[0]]
                essential_data.pop(command.id_column.columns[0])
            
            business_embedding = BusinessEmbeddings(
                id=id,
                embeddings=embedding,
                metadata=essential_data,
            )

            embeddings_batch.add_embeddings(business_embedding)

        self.embeddings_repository.update_batch(embeddings_batch)
