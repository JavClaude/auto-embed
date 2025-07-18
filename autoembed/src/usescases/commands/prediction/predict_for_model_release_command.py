from dataclasses import dataclass

from autoembed.src.yaml.auto_embed_yaml_schema import IdColumn, Modeling, PredictionData, VectorStore


@dataclass
class PredictForModelReleaseCommand:
    project_name: str
    model_version: str
    id_column: IdColumn
    vector_store: VectorStore
    prediction_data: PredictionData
    modeling: Modeling
