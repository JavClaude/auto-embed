import fire

from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases import (
    WhatIsMyClassifiedRecommendationsUsecases,
)
from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases_query import (
    WhatIsMyClassifiedRecommendationsQuery,
)


def what_is_my_classified_recommendation(classified_ref: str):
    what_is_my_classified_recommendation_usecase = WhatIsMyClassifiedRecommendationsUsecases()
    what_is_my_classified_recommendation_command = WhatIsMyClassifiedRecommendationsQuery(classified_ref)
    recommendations = what_is_my_classified_recommendation_usecase.ask(what_is_my_classified_recommendation_command)

    print(recommendations)


def main():
    fire.Fire(what_is_my_classified_recommendation)
