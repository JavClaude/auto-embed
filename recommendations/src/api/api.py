import uvicorn
from fastapi import FastAPI

from recommendations.src.domain.entites.classified_events import ClassifiedEventType
from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases import WhatIsMyClassifiedRecommendationsUsecases
from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases_query import WhatIsMyClassifiedRecommendationsQuery
from recommendations.src.usescases.users.update_users_embedding_usecase import UpdateUsersEmbeddingUsecase

app = FastAPI()


what_is_my_classified_recommendations_usecases = WhatIsMyClassifiedRecommendationsUsecases()
update_users_embedding_usecase = UpdateUsersEmbeddingUsecase()

@app.get("/health")
def health():
    return {"message": "OK"}


@app.get("/api/v1/classified/{classified_ref}/recommendations")
def get_recommendations(classified_ref: str):
    query = WhatIsMyClassifiedRecommendationsQuery(classified_ref)
    recommendations = what_is_my_classified_recommendations_usecases.ask(query)
    return {
        "classified_ref": classified_ref,
        "recommendations": recommendations,
    }


@app.post("/api/v1/users/{user_id}/classified/{classified_ref}/update-embedding")
def update_user_embedding(user_id: str, classified_ref: str, classified_event_type: ClassifiedEventType):
    update_users_embedding_usecase.execute(user_id, classified_ref, classified_event_type)


@app.get("/api/v1/users/{user_id}/recommendations")
def get_user_recommendations(user_id: str):
    # Create usecase
    return {
        "user_id": user_id,
        "recommendations": recommendations,
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)
