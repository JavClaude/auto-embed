import uvicorn
from fastapi import FastAPI

from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases import WhatIsMyClassifiedRecommendationsUsecases
from recommendations.src.usescases.queries.what_is_my_classified_recommendations_usecases_query import WhatIsMyClassifiedRecommendationsQuery

app = FastAPI()

@app.get("/health")
def read_root():
    return {"message": "OK"}

@app.get("/api/v1/classified/{classified_ref}/recommendations")
def get_recommendations(classified_ref: str):
    query = WhatIsMyClassifiedRecommendationsQuery(classified_ref)
    recommendations = WhatIsMyClassifiedRecommendationsUsecases().ask(query)
    return {
        "classified_ref": classified_ref,
        "recommendations": recommendations,
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)
