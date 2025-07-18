from dataclasses import dataclass

from autoembed.src.yaml.auto_embed_yaml_schema import Modeling, TrainingData, VectorStore


@dataclass
class TrainEmbeddingModelCommand:
    project_name: str
    vector_store: VectorStore
    training_data: TrainingData
    modeling: Modeling