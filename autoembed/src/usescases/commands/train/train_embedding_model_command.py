from dataclasses import dataclass

from autoembed.src.yaml.auto_embed_yaml_schema import Modeling, TrainingData, VectorStore


@dataclass
class TrainEmbeddingModelCommand:
    model_name: str
    id_column: str
    vector_store: VectorStore
    training_data: TrainingData
    modeling: Modeling