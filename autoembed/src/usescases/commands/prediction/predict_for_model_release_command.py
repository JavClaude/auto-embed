from dataclasses import dataclass

from autoembed.src.domain.entities.vector_store_specs import VectorStoreSpecs
from autoembed.src.yaml.auto_embed_yaml_schema import Modeling, PredictionData


@dataclass
class PredictForModelReleaseCommand:
    project_name: str
    model_version: str
    vector_store: VectorStoreSpecs
    prediction_data: PredictionData
    modeling: Modeling
