from logging import Logger

from kink import inject
from autoembed.src.domain.dataset_preprocessor import DatasetPreprocessor
from autoembed.src.domain.interfaces.embedding_model_interface import EmbeddingModelInterface
from autoembed.src.domain.interfaces.embeddings_repository_interface import EmbeddingsRepositoryInterface
from autoembed.src.domain.interfaces.data_repository_interface import DataRepositoryInterface
from autoembed.src.domain.interfaces.model_registry_interface import ModelRegistryInterface
from autoembed.src.domain.services.batch_embeddings_service import BatchBusinessEmbeddingService
from autoembed.src.usescases.commands.prediction.embed_text_column_for_model_release_command import EmbedTextColumnForModelReleaseCommand


@inject()
class EmbedTextColumnForModelReleaseUsecase:
    def __init__(self, data_repository: DataRepositoryInterface, model_registry: ModelRegistryInterface, embedding_model: EmbeddingModelInterface, embeddings_repository: EmbeddingsRepositoryInterface, batch_business_embedding_service: BatchBusinessEmbeddingService, logger: Logger):
        self.data_repository = data_repository
        self.model_registry = model_registry
        self.embedding_model = embedding_model
        self.embeddings_repository = embeddings_repository
        self.batch_business_embedding_service = batch_business_embedding_service
        self.logger = logger
        
    def execute(self, command: EmbedTextColumnForModelReleaseCommand):
        self.logger.info(f"Embedding text column: {command.modeling.modeling_columns.text_column} for model release {command.project_name} {command.model_version} for {command.prediction_data.path}")

        prediction_data = self.data_repository.get_prediction_data(command.prediction_data.path)
        preprocessor_json = self.model_registry.load_json_preprocessor(command.project_name, command.model_version)
        dataset_preprocessor = DatasetPreprocessor.from_dict_definition(preprocessor_json)
        model = self.model_registry.load_model(self.embedding_model, command.project_name, command.model_version)

        preprocessed_data = dataset_preprocessor.preprocess(prediction_data)
        embeddings = model.embed_text_column(preprocessed_data)

        self.logger.info(f"Text column embeddings: {embeddings.shape}")
