from logging import Logger

from kink import inject

from recommendations.src.domain.interfaces.classified_embedding_model_interface import (
    ClassifiedEmbeddingModelInterface,
)
from recommendations.src.domain.interfaces.classified_embeddings_repository_interface import (
    ClassifiedEmbeddingsBatch,
    ClassifiedEmbeddingsRepositoryInterface,
)
from recommendations.src.domain.interfaces.classified_repository_interface import (
    ClassifiedRepositoryInterface,
)
from recommendations.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)
from recommendations.src.usescases.commands.prediction.predict_for_model_release_command import (
    PredictForModelReleaseCommand,
)


@inject()
class PredictForModelReleaseUsecase:
    def __init__(
        self,
        classified_repository: ClassifiedRepositoryInterface,
        classified_embedding_model_registry: ModelRegistryInterface,
        classified_embedding_model: ClassifiedEmbeddingModelInterface,
        logger: Logger,
        classified_embeddings_repository: ClassifiedEmbeddingsRepositoryInterface,
    ):
        self.logger = logger
        self.classified_repository = classified_repository
        self.classified_embedding_model_registry = classified_embedding_model_registry
        self.classified_embeddings_repository = classified_embeddings_repository
        self.classified_embedding_model = classified_embedding_model

    def execute(self, command: PredictForModelReleaseCommand) -> None:
        self.logger.info(f"Predicting for model release {command.model_id} for {command.date_to_predict}")

        prediction_data = self.classified_repository.get_classified_prediction_data(command.date_to_predict)
        dataset_preprocessor = self.classified_embedding_model_registry.load_preprocessor(command.model_id)
        model = self.classified_embedding_model_registry.load_model(self.classified_embedding_model, command.model_id)

        preprocessed_classified_data = dataset_preprocessor.preprocess(prediction_data)
        classified_embeddings = model.embed(preprocessed_classified_data)

        classified_embeddings_batch = ClassifiedEmbeddingsBatch(
            classified_refs=prediction_data["classified_ref"].tolist(),
            classified_embeddings=classified_embeddings,
        )

        self.logger.info(f"Updating {len(classified_embeddings_batch.classified_refs)} classified embeddings")
        self.classified_embeddings_repository.update_batch(classified_embeddings_batch)
