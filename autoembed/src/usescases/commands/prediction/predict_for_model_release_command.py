from dataclasses import dataclass

from autoembed.src.yaml.auto_embed_yaml_schema import IdColumns, Modeling, PredictionData, VectorStore


@dataclass
class PredictForModelReleaseCommand:
    project_name: str
    model_version: str
    id_column: IdColumns
    vector_store: VectorStore
    prediction_data: PredictionData
    modeling: Modeling
