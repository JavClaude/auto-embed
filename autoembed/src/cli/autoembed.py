from enum import Enum
from logging import Logger

import yaml
import typer
from kink import di, inject

from autoembed.src.domain.interfaces.embeddings_repository_interface import EmbeddingsRepositoryInterface
from autoembed.src.infrastructure.embeddings.embedding_chromadb_adapter import EmbeddingsChromaDbAdapter
from autoembed.src.usescases.commands.visualize.generate_interactive_visualization_command import GenerateInteractiveVisualizationCommand
from autoembed.src.usescases.commands.visualize.generate_interactive_visualization_command_usecase import GenerateInteractiveVisualizationCommandUsecase
from autoembed.src.yaml.auto_embed_yaml_schema import AutoEmbedByYamlFileSchema
from autoembed.src.usescases.commands.prediction.predict_for_model_release_command import PredictForModelReleaseCommand
from autoembed.src.usescases.commands.prediction.predict_for_model_release_usecase import PredictForModelReleaseUsecase
from autoembed.src.usescases.commands.train.train_embedding_model_command import TrainEmbeddingModelCommand
from autoembed.src.usescases.commands.train.train_embeddings_model_usecase import TrainEmbeddingModelUseCase


class AutoEmbedMode(str, Enum):
    TRAIN = "train"
    PREDICT = "predict"
    # EMBED_TEXT_COLUMN = "embed-text-column"
    # ASK_TEXTUAL_RECOMMENDATIONS = "ask-textual-recommendations"
    SERVE = "serve"
    VISUALIZE = "visualize"

@inject()
def autoembed(mode: AutoEmbedMode, yaml_path: str, text: str | None = None, project_name: str | None = None, model_version: str | None = None):
    with open(yaml_path, "r") as f:
        yaml_as_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Build bus middleware from yaml

    auto_embed_yaml_schema = AutoEmbedByYamlFileSchema.from_yaml_as_dict(yaml_as_dict)
    logger = di[Logger]
    logger.info(f"Executing command: {mode} with parameters: {auto_embed_yaml_schema}")

    di[EmbeddingsRepositoryInterface] = EmbeddingsChromaDbAdapter(vector_collection_name=auto_embed_yaml_schema.vector_store.vector_collection_name)

    if mode == AutoEmbedMode.TRAIN:
        if auto_embed_yaml_schema.data.training is None:
            raise ValueError(f"Training data is required for mode: {mode}")

        command = TrainEmbeddingModelCommand(
            project_name=auto_embed_yaml_schema.project_name,
            training_data=auto_embed_yaml_schema.data.training,
            modeling=auto_embed_yaml_schema.modeling,
        )
        usecase = TrainEmbeddingModelUseCase()
        usecase.execute(command)

    elif mode == AutoEmbedMode.PREDICT:
        if auto_embed_yaml_schema.data.prediction is None:
            raise ValueError(f"Prediction data is required for mode: {mode}")

        command = PredictForModelReleaseCommand(
            project_name=auto_embed_yaml_schema.project_name,
            model_version=auto_embed_yaml_schema.modeling.model_version,
            vector_store=auto_embed_yaml_schema.vector_store,
            prediction_data=auto_embed_yaml_schema.data.prediction,
            modeling=auto_embed_yaml_schema.modeling,
        )
        usecase = PredictForModelReleaseUsecase()
        usecase.execute(command)

    # elif mode == AutoEmbedMode.EMBED_TEXT_COLUMN:
    #     di[EmbeddingsRepositoryInterface] = EmbeddingsChromaDbAdapter(vector_collection_name=auto_embed_yaml_schema.vector_store.vector_collection_name + "_text_column")

    #     if auto_embed_yaml_schema.data.prediction is None:
    #         raise ValueError(f"Prediction data is required for mode: {mode}")

    #    command = EmbedTextColumnForModelReleaseCommand(
    #         project_name=auto_embed_yaml_schema.project_name,
    #         model_version=auto_embed_yaml_schema.modeling.model_version,
    #         vector_store=auto_embed_yaml_schema.vector_store,
    #         prediction_data=auto_embed_yaml_schema.data.prediction,
    #         modeling=auto_embed_yaml_schema.modeling,
    #     )
    #     usecase = EmbedTextColumnForModelReleaseUsecase()
    #     usecase.execute(command)

    # elif mode == AutoEmbedMode.ASK_TEXTUAL_RECOMMENDATIONS:
        # di[EmbeddingsRepositoryInterface] = EmbeddingsChromaDbAdapter(vector_collection_name=auto_embed_yaml_schema.vector_store.vector_collection_name + "_text_column")

    #     query = WhatIsMyTextualRecommendationsQuery(
    #         text=text,
    #         project_name=project_name,
    #         model_version=model_version,
    #     )
    #     usecase = WhatIsMyTextualRecommendationsUsecases()
    #     most_similar_ids = usecase.ask(query)
    #     logger.info(f"Most similar ids: {most_similar_ids}")

    elif mode == AutoEmbedMode.SERVE:
        logger.warning("Serve mode not implemented yet")
        pass

    elif mode == AutoEmbedMode.VISUALIZE:
        # di[EmbeddingsRepositoryInterface] = EmbeddingsChromaDbAdapter(vector_collection_name=auto_embed_yaml_schema.vector_store.vector_collection_name + "_text_column")
        logger.info(f"Generating interactive visualization for {auto_embed_yaml_schema.visualisation.n_samples} samples")

        command = GenerateInteractiveVisualizationCommand(
            n_samples=auto_embed_yaml_schema.visualisation.n_samples,
            visualisation_columns=auto_embed_yaml_schema.visualisation.visualisation_columns,
        )

        usecase = GenerateInteractiveVisualizationCommandUsecase()
        usecase.execute(command)


def main():
    typer.run(autoembed)
