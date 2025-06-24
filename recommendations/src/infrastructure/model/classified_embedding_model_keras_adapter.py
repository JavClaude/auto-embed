from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Concatenate,
    Layer,
    LayerNormalization,
)

from recommendations.src.domain.dataset_preprocessor import (
    NUMERICAL_INPUTS_FEATURES_KEY,
    NUMERICAL_OUTPUTS_KEY,
)
from recommendations.src.domain.interfaces.classified_embedding_model_interface import (
    ClassifiedEmbeddingModelInterface,
)
from recommendations.src.domain.entites.dataset_analysis import DatasetAnalysis


class KerasAutoencoder(ClassifiedEmbeddingModelInterface):
    def __init__(
        self,
        bottleneck_layer_dim: int | None = None,
        hidden_layer_dim: List[int] | None = None,
        autoencoder: Model | None = None,
        encoder: Model | None = None,
    ):
        self.bottleneck_layer_dim = bottleneck_layer_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.autoencoder: Model | None = autoencoder
        self.encoder: Model | None = encoder

    @classmethod
    def from_model(cls, autoencoder: Model, encoder: Model) -> "KerasAutoencoder":
        return cls(autoencoder=autoencoder, encoder=encoder)

    @classmethod
    def from_dataset_analysis(
        cls,
        dataset_analysis: DatasetAnalysis,
        bottleneck_layer_dim: int,
        hidden_layer_dim: List[int],
    ) -> "KerasAutoencoder":
        autoencoder, encoder = cls._build_model(dataset_analysis, bottleneck_layer_dim, hidden_layer_dim)
        return cls(autoencoder=autoencoder, encoder=encoder)

    def fit(
        self,
        x: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int,
    ) -> None:
        self.autoencoder.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True)

    def embed(self, x: pd.DataFrame) -> np.ndarray:
        return self.encoder.predict(x)

    @classmethod
    def _build_model(
        cls,
        dataset_analysis: DatasetAnalysis,
        bottleneck_layer_dim: int,
        hidden_layer_dim: List[int],
    ) -> Tuple[Model, Model]:
        inputs, bottleneck_layer = cls._build_encoder_part(dataset_analysis, bottleneck_layer_dim, hidden_layer_dim)
        outputs = cls._build_decoder_part(bottleneck_layer, dataset_analysis, bottleneck_layer_dim, hidden_layer_dim)

        autoencoder = Model(inputs=inputs, outputs=outputs)
        encoder = Model(inputs=inputs, outputs=bottleneck_layer)

        losses = {}
        loss_weights = {}

        losses[NUMERICAL_OUTPUTS_KEY] = "mse"
        loss_weights[NUMERICAL_OUTPUTS_KEY] = 1.0

        for feature_name in dataset_analysis.categorical_columns.columns:
            losses[f"{feature_name}_outputs"] = "sparse_categorical_crossentropy"
            loss_weights[f"{feature_name}_outputs"] = 1.0

        autoencoder.compile(optimizer=Adam(learning_rate=0.005), loss=losses, loss_weights=loss_weights)

        return autoencoder, encoder

    @classmethod
    def _build_encoder_part(
        cls,
        dataset_analysis: DatasetAnalysis,
        bottleneck_layer_dim: int,
        hidden_layer_dim: List[int],
    ) -> None:
        inputs = {}
        embeddings = []

        numerical_inputs_size = len(dataset_analysis.numerical_columns.columns)
        numerical_inputs_layer = Input(shape=(numerical_inputs_size,), name=NUMERICAL_INPUTS_FEATURES_KEY)

        inputs[NUMERICAL_INPUTS_FEATURES_KEY] = numerical_inputs_layer
        embeddings.append(numerical_inputs_layer)

        for (
            feature_name,
            feature,
        ) in dataset_analysis.categorical_columns.columns.items():
            categorical_input_layer = Input(shape=(1,), name=feature_name)
            inputs[feature_name] = categorical_input_layer

            embedding_layer = Embedding(
                input_dim=len(feature.vocabulary) + 1,
                output_dim=feature.embedding_dim,
                name=f"{feature_name}_embedding",
                
            )(categorical_input_layer)

            embedding_layer = Flatten(name=f"{feature_name}_embedding_flatten")(embedding_layer)
            embeddings.append(embedding_layer)

        all_features_layer = Concatenate()(embeddings)

        for index, hidden_layer_dim in enumerate(hidden_layer_dim):
            all_features_layer = LayerNormalization()(all_features_layer)
            all_features_layer = Dense(units=hidden_layer_dim, activation="relu", name=f"hidden_layer_{index}")(all_features_layer)
            all_features_layer = LayerNormalization()(all_features_layer)
            all_features_layer = Dropout(0.2)(all_features_layer)

        bottleneck_layer = Dense(units=bottleneck_layer_dim, activation="tanh", name="bottleneck_layer")(all_features_layer)

        return inputs, bottleneck_layer

    @classmethod
    def _build_decoder_part(
        cls,
        bottleneck_layer: Layer,
        dataset_analysis: DatasetAnalysis,
        bottleneck_layer_dim: int,
        hidden_layer_dim: List[int],
    ) -> None:
        first_decoding_layer = Dense(units=bottleneck_layer_dim, activation="relu", name="first_decoding_layer")(bottleneck_layer)

        for index, hidden_layer_dim in enumerate(reversed(hidden_layer_dim)):
            first_decoding_layer = Dense(
                units=hidden_layer_dim,
                activation="relu",
                name=f"decoding_layer_{index}",
            )(first_decoding_layer)
            first_decoding_layer = Dropout(0.2)(first_decoding_layer)

        outputs = {}

        numerical_outputs = Dense(
            units=len(dataset_analysis.numerical_columns.columns),
            name="numerical_outputs",
        )(first_decoding_layer)
        outputs[NUMERICAL_OUTPUTS_KEY] = numerical_outputs

        for (
            feature_name,
            feature,
        ) in dataset_analysis.categorical_columns.columns.items():
            categorical_output_layer = Dense(
                units=len(feature.vocabulary) + 1,
                name=f"{feature_name}_outputs",
                activation="softmax",
            )(first_decoding_layer)
            outputs[f"{feature_name}_outputs"] = categorical_output_layer

        return outputs

    def save(self, path: str) -> None:
        self.autoencoder.save(f"{path}/autoencoder.keras")
        self.encoder.save(f"{path}/encoder.keras")

    @classmethod
    def load(cls, path: str) -> "KerasAutoencoder":
        autoencoder = load_model(f"{path}/autoencoder.keras")
        encoder = load_model(f"{path}/encoder.keras")
        return cls.from_model(autoencoder, encoder)
