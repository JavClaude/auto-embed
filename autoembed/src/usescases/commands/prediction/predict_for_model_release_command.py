from dataclasses import dataclass

from autoembed.src.yaml.auto_embed_yaml_schema import Modeling, PredictionData, VectorStore


@dataclass
class PredictForModelReleaseCommand:
    model_name: str
    id_column: str
    vector_store: VectorStore
    prediction_data: PredictionData
    modeling: Modeling
