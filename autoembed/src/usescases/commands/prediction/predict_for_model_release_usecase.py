from logging import Logger

from kink import inject

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
from autoembed.src.domain.services.batch_embeddings_service import BatchBusinessEmbeddingService
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
        embeddings_repository: EmbeddingsRepositoryInterface,
        batch_business_embedding_service: BatchBusinessEmbeddingService,
        logger: Logger,
    ):
        self.logger = logger
        self.data_repository = data_repository
        self.model_registry = model_registry
        self.embeddings_repository = embeddings_repository
        self.embedding_model = embedding_model
        self.batch_business_embedding_service = batch_business_embedding_service

    def execute(self, command: PredictForModelReleaseCommand) -> None:
        self.logger.info(f"Predicting for model release {command.project_name} {command.model_version} for {command.prediction_data.path}")

        prediction_data = self.data_repository.get_prediction_data(command.prediction_data.path)
        dataset_preprocessor = self.model_registry.load_preprocessor(command.project_name, command.model_version)
        model = self.model_registry.load_model(self.embedding_model, command.project_name, command.model_version)

        preprocessed_data = dataset_preprocessor.preprocess(prediction_data)
        embeddings = model.embed(preprocessed_data)

        embeddings_batch = self.batch_business_embedding_service.generate_batch_business_embeddings(command.id_column, command.vector_store.metadata_columns, embeddings, prediction_data)

        self.embeddings_repository.update_batch(embeddings_batch)
