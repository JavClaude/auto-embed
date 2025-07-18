import datetime
from logging import Logger
import uuid

from kink import inject

from autoembed.src.domain.dataset_preprocessor import (
    DatasetPreprocessor,
)
from autoembed.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)
from autoembed.src.domain.interfaces.data_repository_interface import (
    DataRepositoryInterface,
)
from autoembed.src.infrastructure.model.embedding_model_keras_adapter import KerasAutoencoder
from autoembed.src.usescases.commands.train.train_embedding_model_command import (
    TrainEmbeddingModelCommand,
)


@inject()
class TrainEmbeddingModelUseCase:
    def __init__(
        self,
        data_repository: DataRepositoryInterface,
        embedding_model_registry: ModelRegistryInterface,
        logger: Logger,
    ):
        self.data_repository = data_repository
        self.embedding_model_registry = embedding_model_registry
        self.logger = logger

    def execute(self, command: TrainEmbeddingModelCommand) -> None:
        self.logger.info(f"✅ Training embeddings model with parameters: {command}")

        training_data = self.data_repository.get_training_data(command.training_data.path)

        if command.modeling.light_mode:
            self.logger.info("✅ Sampling training data for light mode")
            if len(training_data) > command.modeling.light_mode_sample_size:
                training_data = training_data.sample(n=command.modeling.light_mode_sample_size)
            else:
                self.logger.warning(f"⚠️ Training data is less than {command.modeling.light_mode_sample_size}, using all data ({len(training_data)})")

        self.logger.info("🔍 Fitting dataset preprocessor")

        dataset_preprocessor = DatasetPreprocessor(command.modeling.modeling_columns.numerical_columns, command.modeling.modeling_columns.categorical_columns)
        dataset_preprocessor.fit(training_data)

        preprocessed_data = dataset_preprocessor.preprocess(training_data)
        preprocessed_target = dataset_preprocessor.preprocess_target(training_data)

        self.logger.info("🔍 Fitting embeddings model")
        dataset_analysis = dataset_preprocessor.get_analysis()

        model = KerasAutoencoder.from_dataset_analysis(
            dataset_analysis,
            command.modeling.bottle_neck_size, 
            command.modeling.hidden_layer_sizes,
        )

        self.logger.info(f"🔍 Numerical columns: {len(dataset_analysis.numerical_columns.columns)}")
        self.logger.info(f"🔍 Categorical columns: {len(dataset_analysis.categorical_columns.columns)}")

        model.fit(
            preprocessed_data,
            preprocessed_target,
            epochs=command.modeling.epochs,
            batch_size=command.modeling.batch_size,
        )
        
        self.embedding_model_registry.save_model_and_preprocessor(model, dataset_preprocessor, command.project_name)