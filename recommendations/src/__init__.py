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
from recommendations.src.domain.interfaces.user_embedding_repository_interface import UserEmbeddingRepositoryInterface
from recommendations.src.infrastructure.embeddings.classified_embedding_chromadb_adapter import (
    ClassifiedEmbeddingsChromaDbAdapter,
)
from recommendations.src.infrastructure.embeddings.user_embedding_repository_chromadb_adapter import UserEmbeddingRepositoryChromaDbAdapter
from recommendations.src.infrastructure.model.classified_embedding_model_keras_adapter import (
    KerasAutoencoder,
)
from recommendations.src.infrastructure.classified_repository.classified_repository_local_adapter import (
    ClassifiedRepositoryLocalAdapter,
)
from recommendations.src.infrastructure.model.local_model_registry_adapter import (
    LocalModelRegistryAdapter,
)
from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases import WhatIsMyClassifiedRecommendationsUsecases

# Infrastructure layer
di[Logger] = logger

di[ClassifiedRepositoryInterface] = ClassifiedRepositoryLocalAdapter(path="data")
di[ModelRegistryInterface] = LocalModelRegistryAdapter(path="models")
di[ClassifiedEmbeddingsRepositoryInterface] = ClassifiedEmbeddingsChromaDbAdapter()
di[UserEmbeddingRepositoryInterface] = UserEmbeddingRepositoryChromaDbAdapter()
di[ClassifiedEmbeddingModelInterface] = KerasAutoencoder

di[WhatIsMyClassifiedRecommendationsUsecases] = WhatIsMyClassifiedRecommendationsUsecases()
