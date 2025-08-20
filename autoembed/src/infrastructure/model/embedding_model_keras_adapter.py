from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate, Layer, LayerNormalization, GlobalAveragePooling1D
from keras_hub.layers import TransformerEncoder

from autoembed.src.domain.dataset_preprocessor import (
    NUMERICAL_INPUTS_FEATURES_KEY,
    NUMERICAL_OUTPUTS_KEY,
)
from autoembed.src.domain.interfaces.embedding_model_interface import (
    EmbeddingModelInterface,
)
from autoembed.src.domain.models.dataset_analysis import DatasetAnalysis
from autoembed.src.domain.models.preprocessed_data import PreprocessedData
from autoembed.src.domain.models.preprocessed_target import PreprocessedTarget


class KerasAutoencoder(EmbeddingModelInterface):
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
        x: PreprocessedData,
        y: PreprocessedTarget,
        epochs: int,
        batch_size: int,
    ) -> None:

        self.autoencoder.fit(
            x.to_dict(), y.to_dict(), epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
        )

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

        if dataset_analysis.numerical_columns is not None:
            losses[NUMERICAL_OUTPUTS_KEY] = "mse"
            loss_weights[NUMERICAL_OUTPUTS_KEY] = 1.0

        if dataset_analysis.categorical_columns is not None:
            for feature_name in dataset_analysis.categorical_columns.columns.keys():
                if dataset_analysis.categorical_features_loss_weights is not None:
                    loss_weights[f"{feature_name}_outputs"] = dataset_analysis.categorical_features_loss_weights[feature_name]
                else:
                    loss_weights[f"{feature_name}_outputs"] = 1.0
                losses[f"{feature_name}_outputs"] = "sparse_categorical_crossentropy"

        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=losses, loss_weights=loss_weights)

        return autoencoder, encoder

    @classmethod
    def _build_encoder_part(
        cls,
        dataset_analysis: DatasetAnalysis,
        bottleneck_layer_dim: int,
        hidden_layer_dim: List[int],
    ) -> Tuple[dict, Layer]:
        inputs = {}
        embeddings = []

        if dataset_analysis.numerical_columns is not None:
            numerical_inputs_size = len(dataset_analysis.numerical_columns.columns)
            numerical_inputs_layer = Input(shape=(numerical_inputs_size,), name=NUMERICAL_INPUTS_FEATURES_KEY)

            inputs[NUMERICAL_INPUTS_FEATURES_KEY] = numerical_inputs_layer
            embeddings.append(numerical_inputs_layer)

        if dataset_analysis.categorical_columns is not None:
            for (
                feature_name,
                feature,
            ) in dataset_analysis.categorical_columns.columns.items():
                categorical_input_layer = Input(shape=(1,), name=f"{feature_name}_inputs")
                inputs[f"{feature_name}_inputs"] = categorical_input_layer

                embedding_layer = Embedding(
                    input_dim=len(feature.vocabulary),
                    output_dim=feature.embedding_dim,
                    name=f"{feature_name}_embedding",
                )(categorical_input_layer)

                embedding_layer = Flatten(name=f"{feature_name}_embedding_flatten")(embedding_layer)
                embeddings.append(embedding_layer)

        if dataset_analysis.text_column is not None:
            text_input_layer = Input(shape=(dataset_analysis.text_column.max_length,), name=f"{dataset_analysis.text_column.name}_text_input")
            inputs[f"{dataset_analysis.text_column.name}_text_input"] = text_input_layer

            text_embedding = Embedding(
                input_dim=dataset_analysis.text_column.vocab_size, output_dim=dataset_analysis.text_column.word_embedding, mask_zero=True, name=f"{dataset_analysis.text_column.name}_token_embedding"
            )(text_input_layer)

            # TODO: Ajouter le nombre de layers en param
            transformer_encoder = TransformerEncoder(
                num_heads=4,
                intermediate_dim=dataset_analysis.text_column.word_embedding * 2,
                dropout=0.1,
                activation="relu",
                layer_norm_epsilon=1e-6,
                name=f"{dataset_analysis.text_column.name}_transformer_encoder",
            )(text_embedding)

            text_encoded = GlobalAveragePooling1D(name=f"{dataset_analysis.text_column.name}_pooling")(transformer_encoder)

            embeddings.append(text_encoded)

        if len(embeddings) > 1:
            all_features_layer = Concatenate(name="concatenate_all_features")(embeddings)
        else:
            all_features_layer = embeddings[0]

        for index, hidden_dim in enumerate(hidden_layer_dim):
            all_features_layer = LayerNormalization(name=f"encoder_layer_norm_{index}")(all_features_layer)
            all_features_layer = Dense(units=hidden_dim, activation="leaky_relu", name=f"encoder_hidden_layer_{index}")(all_features_layer)
            all_features_layer = LayerNormalization(name=f"encoder_layer_norm_post_{index}")(all_features_layer)
            all_features_layer = Dropout(0.2, name=f"encoder_dropout_{index}")(all_features_layer)

        bottleneck_layer = Dense(units=bottleneck_layer_dim, activation="tanh", name="bottleneck_layer")(all_features_layer)

        return inputs, bottleneck_layer

    @classmethod
    def _build_decoder_part(
        cls,
        bottleneck_layer: Layer,
        dataset_analysis: DatasetAnalysis,
        bottleneck_layer_dim: int,
        hidden_layer_dim: List[int],
    ) -> dict:

        first_decoding_layer = Dense(units=bottleneck_layer_dim, activation="leaky_relu", name="first_decoding_layer")(bottleneck_layer)

        for index, hidden_dim in enumerate(reversed(hidden_layer_dim)):
            first_decoding_layer = Dense(
                units=hidden_dim,
                activation="relu",
                name=f"decoding_layer_{index}",
            )(first_decoding_layer)
            first_decoding_layer = Dropout(0.2, name=f"decoder_dropout_{index}")(first_decoding_layer)

        outputs = {}

        if dataset_analysis.numerical_columns is not None:
            numerical_outputs = Dense(
                units=len(dataset_analysis.numerical_columns.columns),
                name=NUMERICAL_OUTPUTS_KEY,
            )(first_decoding_layer)
            outputs[NUMERICAL_OUTPUTS_KEY] = numerical_outputs

        if dataset_analysis.categorical_columns is not None:
            for (
                feature_name,
                feature,
            ) in dataset_analysis.categorical_columns.columns.items():
                categorical_output_layer = Dense(
                    units=len(feature.vocabulary),
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
