import pandas as pd
import pytest

from autoembed.src.domain.dataset_preprocessor import DatasetPreprocessor


def test_build_and_fit_dataset_preprocessor_with_numerical_and_categorical_columns():
    # Given
    data = pd.DataFrame(
        data={
            "city": ["Paris", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Ile-de-France", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2025, 2026],
            "height": [170, 180, 190, 200],
        }
    )

    # When
    dataset_preprocessor = DatasetPreprocessor(numerical_columns_names=["income", "year"], categorical_columns_names=["city", "region"])
    dataset_preprocessor.fit(data)

    # Then
    assert len(dataset_preprocessor.numerical_columns.columns) == 2
    assert dataset_preprocessor.numerical_columns.columns["income"].name == "income"
    assert dataset_preprocessor.numerical_columns.columns["year"].name == "year"

    assert len(dataset_preprocessor.categorical_columns.columns) == 2
    assert dataset_preprocessor.categorical_columns.columns["city"].name == "city"
    assert dataset_preprocessor.categorical_columns.columns["region"].name == "region"

    # TODO: Add a dedicated unit test for the categorical features loss weights
    assert dataset_preprocessor.categorical_features_loss_weights == {"city": 1.2618595071429148, "region": 1.0}


def test_build_and_fit_dataset_preprocessor_with_numerical_columns_only():
    # Given
    data = pd.DataFrame(
        data={
            "city": ["Paris", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Ile-de-France", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2025, 2026],
            "height": [170, 180, 190, 200],
        }
    )

    # When
    dataset_preprocessor = DatasetPreprocessor(numerical_columns_names=["income", "year"])
    dataset_preprocessor.fit(data)

    # Then
    assert len(dataset_preprocessor.numerical_columns.columns) == 2
    assert dataset_preprocessor.numerical_columns.columns["income"].name == "income"
    assert dataset_preprocessor.numerical_columns.columns["year"].name == "year"

    assert not dataset_preprocessor.categorical_columns

    # TODO: Add a dedicated unit test for the categorical features loss weights
    assert not dataset_preprocessor.categorical_features_loss_weights


def test_build_and_fit_dataset_preprocessor_with_categorical_columns_only():
    # Given
    data = pd.DataFrame(
        data={
            "city": ["Paris", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Ile-de-France", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2025, 2026],
            "height": [170, 180, 190, 200],
        }
    )

    # When
    dataset_preprocessor = DatasetPreprocessor(categorical_columns_names=["city", "region"])
    dataset_preprocessor.fit(data)

    # Then
    assert len(dataset_preprocessor.categorical_columns.columns) == 2
    assert dataset_preprocessor.categorical_columns.columns["city"].name == "city"
    assert dataset_preprocessor.categorical_columns.columns["region"].name == "region"

    assert not dataset_preprocessor.numerical_columns

    # TODO: Add a dedicated unit test for the categorical features loss weights
    assert dataset_preprocessor.categorical_features_loss_weights == {"city": 1.2618595071429148, "region": 1.0}


def test_build_dataset_preprocessor_with_no_columns_should_raise_error():
    # Given / When
    with pytest.raises(ValueError) as exc_info:
        _ = DatasetPreprocessor()

    # Then
    assert exc_info.value.args[0] == "numerical_columns_names or categorical_columns_names or numerical_columns or categorical_columns must be provided"


def test_preprocess_dataset_preprocessor_with_numerical_and_categorical_columns():
    # Given
    training_data = pd.DataFrame(
        data={
            "city": ["Paris", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Ile-de-France", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2025, 2026],
            "height": [170, 180, 190, 200],
        }
    )

    prediction_data = pd.DataFrame(
        data={
            "city": ["toulouse", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Midi-Pyrénées", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2026, 2027],
            "height": [170, 180, 200, 210],
        }
    )

    dataset_preprocessor = DatasetPreprocessor(numerical_columns_names=["income", "year"], categorical_columns_names=["city", "region"])
    dataset_preprocessor.fit(training_data)

    # When
    transformed_data = dataset_preprocessor.preprocess(prediction_data)
    transformed_target = dataset_preprocessor.preprocess_target(prediction_data)

    # Then
    assert len(transformed_data) == 3
    assert transformed_data["numerical_inputs_features"].shape == (4, 2)
    assert transformed_data["city"].shape == (4,)
    assert transformed_data["region"].shape == (4,)

    assert len(transformed_target) == 3
    assert transformed_target["numerical_outputs"].shape == (4, 2)
    assert transformed_target["city_outputs"].shape == (4,)
    assert transformed_target["region_outputs"].shape == (4,)


def test_preprocess_dataset_preprocessor_with_numerical_columns_only():
    # Given
    training_data = pd.DataFrame(
        data={
            "city": ["Paris", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Ile-de-France", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2025, 2026],
            "height": [170, 180, 190, 200],
        }
    )

    prediction_data = pd.DataFrame(
        data={
            "city": ["toulouse", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Midi-Pyrénées", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2026, 2027],
            "height": [170, 180, 200, 210],
        }
    )

    dataset_preprocessor = DatasetPreprocessor(numerical_columns_names=["income", "year"])
    dataset_preprocessor.fit(training_data)

    # When
    transformed_data = dataset_preprocessor.preprocess(prediction_data)
    transformed_target = dataset_preprocessor.preprocess_target(prediction_data)

    # Then
    assert len(transformed_data) == 1
    assert transformed_data["numerical_inputs_features"].shape == (4, 2)

    assert len(transformed_target) == 1
    assert transformed_target["numerical_outputs"].shape == (4, 2)


def test_preprocess_dataset_preprocessor_with_categorical_columns_only():
    # Given
    training_data = pd.DataFrame(
        data={
            "city": ["Paris", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Ile-de-France", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2025, 2026],
            "height": [170, 180, 190, 200],
        }
    )

    prediction_data = pd.DataFrame(
        data={
            "city": ["toulouse", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Midi-Pyrénées", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "income": [10000, 20000, pd.NA, 40000],
            "year": [2022, 2024, 2026, 2027],
            "height": [170, 180, 200, 210],
        }
    )

    dataset_preprocessor = DatasetPreprocessor(categorical_columns_names=["city", "region"])
    dataset_preprocessor.fit(training_data)

    # When
    transformed_data = dataset_preprocessor.preprocess(prediction_data)
    transformed_target = dataset_preprocessor.preprocess_target(prediction_data)

    # Then
    assert len(transformed_data) == 2
    assert transformed_data["city"].shape == (4,)
    assert transformed_data["region"].shape == (4,)

    assert len(transformed_target) == 2
    assert transformed_target["city_outputs"].shape == (4,)
    assert transformed_target["region_outputs"].shape == (4,)
