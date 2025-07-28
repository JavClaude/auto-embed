import pandas as pd
from autoembed.src.domain.columns.categorical.categorical_column import CategoricalColumn
from autoembed.src.domain.columns.categorical.categorical_columns import CategoricalColumns
from autoembed.src.domain.columns.numerical.numerical_column import NumericalColumn
from autoembed.src.domain.columns.numerical.numerical_columns import NumericalColumns
from autoembed.src.domain.dataset_preprocessor import DatasetPreprocessor
from autoembed.src.domain.models.dataset_analysis import DatasetAnalysis
from autoembed.src.infrastructure.model.embedding_model_keras_adapter import KerasAutoencoder


def test_when_build_model_from_dataset_analysis_then_should_build_model_with_correct_dimensions():
    # Given
    numerical_income_column = NumericalColumn(name="income", value_used_to_fill_na=0, mean=0, std=1)
    numerical_year_column = NumericalColumn(name="year", value_used_to_fill_na=0, mean=0, std=1)

    categorical_city_column = CategoricalColumn(
        name="city",
        value_used_to_fill_na="unknown",
        vocabulary={
            "paris": 0,
            "lyon": 1,
            "marseille": 2,
            "toulouse": 3,
            "unknown": 4,
        },
        embedding_dim=12,
    )
    categorical_region_column = CategoricalColumn(
        name="region",
        value_used_to_fill_na="unknown",
        vocabulary={
            "ile-de-france": 0,
            "auvergne-rhone-alpes": 1,
            "midi-pyrenees": 2,
            "unknown": 3,
        },
        embedding_dim=12,
    )

    dataset_analysis = DatasetAnalysis(
        numerical_columns=NumericalColumns.from_numerical_columns([numerical_income_column, numerical_year_column]),
        categorical_columns=CategoricalColumns.from_categorical_columns([categorical_city_column, categorical_region_column]),
        categorical_features_loss_weights={
            "city": 1.0,
            "region": 1.0,
        },
    )

    # When
    model = KerasAutoencoder.from_dataset_analysis(dataset_analysis, bottleneck_layer_dim=2, hidden_layer_dim=[10])

    # Then
    assert model.autoencoder.get_layer("numerical_inputs_features").get_config()["batch_shape"] == (None, 2)
    assert model.autoencoder.get_layer("city_embedding").input_dim == 5
    assert model.autoencoder.get_layer("city_embedding").output_dim == 12

    assert model.autoencoder.get_layer("region_embedding").input_dim == 4
    assert model.autoencoder.get_layer("region_embedding").output_dim == 12

    assert model.autoencoder.get_layer("hidden_layer_0").get_config()["units"] == 10
    assert model.autoencoder.get_layer("bottleneck_layer").get_config()["units"] == 2

    assert model.autoencoder.get_layer("decoding_layer_0").get_config()["units"] == 10
    assert model.autoencoder.get_layer("numerical_outputs").get_config()["units"] == 2
    assert model.autoencoder.get_layer("city_outputs").get_config()["units"] == 5
    assert model.autoencoder.get_layer("region_outputs").get_config()["units"] == 4


def test_when_build_model_from_dataset_analysis_then_should_be_able_to_fit_and_embed():
    # Given
    data = pd.DataFrame(
        {
            "income": [10000, 20000, 30000, 40000, 50000],
            "year": [2020, 2021, 2022, 2023, 2024],
            "city": ["paris", "lyon", "marseille", "toulouse", "unknown"],
            "region": ["ile-de-france", "auvergne-rhone-alpes", "midi-pyrenees", "unknown", "unknown"],
        }
    )

    dataset_preprocessor = DatasetPreprocessor(
        numerical_columns_names=["income", "year"],
        categorical_columns_names=["city", "region"],
    )

    dataset_preprocessor.fit(data)

    x = dataset_preprocessor.preprocess(data)
    y = dataset_preprocessor.preprocess_target(data)

    model = KerasAutoencoder.from_dataset_analysis(dataset_preprocessor.get_analysis(), bottleneck_layer_dim=2, hidden_layer_dim=[10])

    model.fit(x, y, epochs=1, batch_size=2)

    # Then
    embedding = model.embed(x)

    assert embedding.shape == (5, 2)

# Add test with no numerical columns
# Add test with no categorical columns