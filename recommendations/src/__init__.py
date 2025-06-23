from logging import Logger
from loguru import logger
from kink import di

from recommendations.src.domain.interfaces.classified_embedding_model_interface import (
    ClassifiedEmbeddingModelInterface,
)
from recommendations.src.domain.interfaces.classified_embeddings_repository_interface import (
    ClassifiedEmbeddingsRepositoryInterface,
)
from recommendations.src.domain.interfaces.classified_repository_interface import (
    ClassifiedRepositoryInterface,
)
from recommendations.src.domain.interfaces.model_registry_interface import (
    ModelRegistryInterface,
)
from recommendations.src.infrastructure.embeddings.classified_embedding_chromadb_adapter import (
    ClassifiedEmbeddingsChromaDbAdapter,
)
from recommendations.src.infrastructure.models.classified_embedding_model_keras_adapter import (
    KerasAutoencoder,
)
from recommendations.src.infrastructure.classified_repository.classified_repository_local_adapter import (
    ClassifiedRepositoryLocalAdapter,
)
from recommendations.src.infrastructure.models.local_model_registry_adapter import (
    LocalModelRegistryAdapter,
)

di[Logger] = logger

di[ClassifiedRepositoryInterface] = ClassifiedRepositoryLocalAdapter(path="data")
di[ModelRegistryInterface] = LocalModelRegistryAdapter(path="models")
di[ClassifiedEmbeddingsRepositoryInterface] = ClassifiedEmbeddingsChromaDbAdapter()
di[ClassifiedEmbeddingModelInterface] = KerasAutoencoder
