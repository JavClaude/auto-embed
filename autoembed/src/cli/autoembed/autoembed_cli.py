from typing import Literal
from logging import Logger

import yaml
import fire
from kink import di

from autoembed.src.domain.interfaces.embeddings_repository_interface import EmbeddingsRepositoryInterface
from autoembed.src.infrastructure.embeddings.embedding_chromadb_adapter import EmbeddingsChromaDbAdapter
from autoembed.src.yaml.auto_embed_yaml_schema import AutoEmbedByYamlFileSchema
from autoembed.src.usescases.commands.prediction.predict_for_model_release_command import PredictForModelReleaseCommand
from autoembed.src.usescases.commands.prediction.predict_for_model_release_usecase import PredictForModelReleaseUsecase
from autoembed.src.usescases.commands.train.train_embedding_model_command import TrainEmbeddingModelCommand
from autoembed.src.usescases.commands.train.train_embeddings_model_usecase import TrainEmbeddingModelUseCase


def autoembed(mode: Literal["train", "predict", "serve", "visualize"], yaml_path: str):
    """CLI principale pour autoembed."""
    
    # Récupération du logger depuis DI
    logger = di[Logger]
    
    # Lecture et validation du fichier YAML de configuration
    with open(yaml_path, "r") as f:
        yaml_as_dict = yaml.load(f, Loader=yaml.FullLoader)

    auto_embed_yaml_schema = AutoEmbedByYamlFileSchema.from_yaml_as_dict(yaml_as_dict)
    logger.info(f"Executing command: {mode} with parameters: {auto_embed_yaml_schema.to_json()}")
    
    # Configuration du repository d'embeddings
    di[EmbeddingsRepositoryInterface] = EmbeddingsChromaDbAdapter(
        vector_collection_name=auto_embed_yaml_schema.vector_store.vector_collection_name
    )

    if mode == "train":
        if auto_embed_yaml_schema.data.training is None:
            raise ValueError(f"Training data is required for mode: {mode}")

        command = TrainEmbeddingModelCommand(
            model_name=auto_embed_yaml_schema.model_name,
            id_column=auto_embed_yaml_schema.id_column,
            vector_store=auto_embed_yaml_schema.vector_store,
            training_data=auto_embed_yaml_schema.data.training,
            modeling=auto_embed_yaml_schema.modeling,
        )
        usecase = TrainEmbeddingModelUseCase()
        usecase.execute(command)

    elif mode == "predict":
        if auto_embed_yaml_schema.data.prediction is None:
            raise ValueError(f"Prediction data is required for mode: {mode}")

        command = PredictForModelReleaseCommand(
            auto_embed_yaml_schema.model_name,
            auto_embed_yaml_schema.id_column,
            auto_embed_yaml_schema.vector_store,
            auto_embed_yaml_schema.data.prediction,
            auto_embed_yaml_schema.modeling,
        )
        usecase = PredictForModelReleaseUsecase()
        usecase.execute(command)
        
    elif mode == "serve":
        logger.warning("Serve mode not implemented yet")
        pass
        
    elif mode == "visualize":
        logger.warning("Visualize mode not implemented yet")
        pass


def main():
    fire.Fire(autoembed)