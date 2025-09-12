import pandas as pd

from autoembed.src.domain.columns.categorical.categorical_column import CategoricalColumn


def test_build_categorical_column_from_series():
    # Given
    series = pd.Series(["Paris", "Paris ", "   Lyon", "Lyon", "Marseille", "Marseille", "Marseille", "Marseille", "Marseille", pd.NA], name="city")

    # When
    categorical_column = CategoricalColumn.from_series(series)

    # Then
    assert categorical_column.name == "city"
    assert categorical_column.vocabulary == {"paris": 0, "lyon": 1, "marseille": 2, "unk": 3}
    assert categorical_column.embedding_dim == 32


def test_transform_categorical_column_with_na_and_unknown_valuuuues():
    # Given
    categorical_column = CategoricalColumn.from_series(pd.Series(["Paris", "Paris ", "   Lyon", "Marseille", "Marseille", pd.NA], name="city"))
    series_to_transform = pd.Series(
        ["Paris", pd.NA, "unknown_values", "lyon"],
    )

    # When
    transformed_series = categorical_column.transform(series_to_transform)

    # Then
    assert transformed_series.to_list() == [0, 3, 3, 1]
