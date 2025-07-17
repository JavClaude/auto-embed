from typing import List

import fire

from recommendations.src.usescases.commands.train.train_classified_embedding_model_command import (
    TrainClassifiedEmbeddingsModelCommand,
)
from recommendations.src.usescases.commands.train.train_classified_embeddings_model_usecase import (
    TrainEmbeddingModelUseCase,
)


def train_classified_embedding_model(
    online_date: str = "2025-06-19",
    bottle_neck_size: int = 96,
    hidden_layer_sizes: List[int] = [512, 256, 128],
    epochs: int = 5,
    batch_size: int = 256,
    light_mode: bool = False,
):
    train_model_usecase = TrainEmbeddingModelUseCase()
    train_model_command = TrainClassifiedEmbeddingsModelCommand(
        online_date,
        bottle_neck_size,
        hidden_layer_sizes,
        epochs,
        batch_size,
        light_mode,
    )
    train_model_usecase.execute(train_model_command)


def main():
    fire.Fire(train_classified_embedding_model)
