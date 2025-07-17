from dataclasses import dataclass
from logging import Logger
from typing import List

from kink import inject

from autoembed.src.domain.interfaces.embeddings_repository_interface import (
    EmbeddingsRepositoryInterface,
)
from autoembed.src.usescases.queries.what_is_my_recommendations_usecases_query import (
    WhatIsMyRecommendationsQuery,
)


@dataclass
@inject()
class WhatIsMyRecommendationsUsecases:
    def __init__(
        self,
        embeddings_repository: EmbeddingsRepositoryInterface,
        logger: Logger,
    ):
        self.embeddings_repository = embeddings_repository
        self.logger = logger

    def ask(self, query: WhatIsMyRecommendationsQuery) -> List[str]:
        self.logger.info(f"Asking for recommendations for {query.id}")
        most_similar_ids = self.embeddings_repository.get_most_similar_embeddings_by_id(query.id)[0]
        return [id for id in most_similar_ids if id != query.id]
