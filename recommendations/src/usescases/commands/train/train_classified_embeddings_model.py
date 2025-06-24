from logging import Logger
import uuid

from kink import inject

from recommendations.src.domain.dataset_preprocessor import (
    CATEGORICAL_COLUMNS_NAMES,
    NUMERICAL_COLUMNS_NAMES,
    DatasetPreprocessor,
)
from recommendations.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)
from recommendations.src.domain.interfaces.classified_repository_interface import (
    ClassifiedRepositoryInterface,
)
from recommendations.src.infrastructure.model.classified_embedding_model_keras_adapter import KerasAutoencoder
from recommendations.src.usescases.commands.train.train_classified_embedding_model_command import (
    TrainClassifiedEmbeddingsModelCommand,
)

TrainClassifiedEmbeddingsModelCommand


@inject()
class TrainClassifiedEmbeddingsModelUsecase:
    def __init__(
        self,
        classified_repository: ClassifiedRepositoryInterface,
        classified_embedding_model_registry: ModelRegistryInterface,
        logger: Logger,
    ):
        self.classified_repository = classified_repository
        self.classified_embedding_model_registry = classified_embedding_model_registry
        self.logger = logger

    def execute(self, command: TrainClassifiedEmbeddingsModelCommand) -> None:
        self.logger.info(f"Training classified embeddings model for {command.online_date}")

        training_data = self.classified_repository.get_classified_training_data(command.online_date)

        if command.light_mode:
            self.logger.info("Sampling training data for light mode")
            training_data = training_data.sample(n=1000)

        self.logger.info("Fitting dataset preprocessor")
        dataset_preprocessor = DatasetPreprocessor(NUMERICAL_COLUMNS_NAMES, CATEGORICAL_COLUMNS_NAMES)
        dataset_preprocessor.fit(training_data)

        preprocessed_data = dataset_preprocessor.preprocess(training_data)
        preprocessed_target = dataset_preprocessor.preprocess_target(training_data)

        self.logger.info("Fitting classified embeddings model")
        dataset_analysis = dataset_preprocessor.get_analysis()

        model = KerasAutoencoder.from_dataset_analysis(dataset_analysis, command.bottle_neck_size, command.hidden_layer_sizes)

        self.logger.info(f"numerical columns: {len(dataset_analysis.numerical_columns.columns)}")
        self.logger.info(f"categorical columns: {len(dataset_analysis.categorical_columns.columns)}")

        model.fit(
            preprocessed_data,
            preprocessed_target,
            epochs=command.epochs,
            batch_size=command.batch_size,
        )

        model_id = f"{command.online_date}-{uuid.uuid4()}-model"
        self.logger.info("Saving classified embeddings model")
        self.classified_embedding_model_registry.save_preprocessor(dataset_preprocessor, model_id)
        self.classified_embedding_model_registry.save_model(model, model_id)
