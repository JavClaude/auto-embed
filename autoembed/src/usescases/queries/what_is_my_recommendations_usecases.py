

from logging import Logger
from typing import List

from kink import inject
from autoembed.src.domain.interfaces.embedding_model_interface import EmbeddingModelInterface
from autoembed.src.domain.interfaces.embeddings_repository_interface import EmbeddingsRepositoryInterface
from autoembed.src.domain.interfaces.model_registry_interface import ModelRegistryInterface
from autoembed.src.usescases.queries.what_is_my_recommendations_usecases_query import WhatIsMyTextualRecommendationsQuery
from autoembed.src.domain.dataset_preprocessor import DatasetPreprocessor


@inject()
class WhatIsMyTextualRecommendationsUsecases:
    def __init__(self, embedding_model: EmbeddingModelInterface, embeddings_repository: EmbeddingsRepositoryInterface, logger: Logger, model_registry: ModelRegistryInterface):
        self.embeddings_repository = embeddings_repository
        self.embedding_model = embedding_model
        self.logger = logger
        self.model_registry = model_registry

    def ask(self, query: WhatIsMyTextualRecommendationsQuery) -> List[str]:
        self.logger.info(f"Asking recommendations for {query.text}")

        preprocessor_json = self.model_registry.load_json_preprocessor(query.project_name, query.model_version)
        dataset_preprocessor = DatasetPreprocessor.from_dict_definition(preprocessor_json)
        model = self.model_registry.load_model(self.embedding_model, query.project_name, query.model_version)

        preprocessed_data = dataset_preprocessor.preprocess_text(query.text)
        embeddings = model.embed_text_column(preprocessed_data)

        most_similar_ids = self.embeddings_repository.get_most_similar_embeddings(embeddings)[0]
        return most_similar_ids