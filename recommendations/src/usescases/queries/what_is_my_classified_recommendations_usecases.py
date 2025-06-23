from dataclasses import dataclass
from logging import Logger
from typing import List

from kink import inject

from recommendations.src.domain.interfaces.classified_embeddings_repository_interface import (
    ClassifiedEmbeddingsRepositoryInterface,
)
from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases_query import (
    WhatIsMyClassifiedRecommendationsQuery,
)


@dataclass
@inject()
class WhatIsMyClassifiedRecommendationsUsecases:
    def __init__(
        self,
        classified_embeddings_repository: ClassifiedEmbeddingsRepositoryInterface,
        logger: Logger,
    ):
        self.classified_embeddings_repository = classified_embeddings_repository
        self.logger = logger

    def ask(self, query: WhatIsMyClassifiedRecommendationsQuery) -> List[str]:
        self.logger.info(f"Asking for recommendations for {query.classified_ref}")

        classified_embeddings = self.classified_embeddings_repository.get_classified_embeddings(query.classified_ref)
        most_similar_classified_embeddings = self.classified_embeddings_repository.get_most_similar_classified_embeddings(classified_embeddings)[0]
        return [classified_ref for classified_ref in most_similar_classified_embeddings if classified_ref != query.classified_ref]
