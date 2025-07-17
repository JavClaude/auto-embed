import warnings

from logging import Logger
from loguru import logger
from kink import di

from autoembed.src.domain.interfaces.embedding_model_interface import (
    EmbeddingModelInterface,
)

from autoembed.src.domain.interfaces.data_repository_interface import (
    DataRepositoryInterface,
)
from autoembed.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)

from autoembed.src.infrastructure.model.embedding_model_keras_adapter import (
    KerasAutoencoder,
)
from autoembed.src.infrastructure.data_repository.data_repository_local_csv_adapter import (
    DataRepositoryLocalCSVAdapter,
)
from autoembed.src.infrastructure.model.local_model_registry_adapter import (
    LocalModelRegistryAdapter,
)

warnings.simplefilter(action='ignore', category=FutureWarning)

di[Logger] = logger

di[DataRepositoryInterface] = DataRepositoryLocalCSVAdapter()
di[ModelRegistryInterface] = LocalModelRegistryAdapter(path="models")
di[EmbeddingModelInterface] = KerasAutoencoder
