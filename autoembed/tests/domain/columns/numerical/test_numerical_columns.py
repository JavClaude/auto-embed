import pandas as pd

from autoembed.src.domain.columns.numerical.numerical_column import NumericalColumn
from autoembed.src.domain.columns.numerical.numerical_columns import NumericalColumns


def test_build_numerical_columns_from_dataframe():
    # Given
    columns_to_keep = ["income", "year"]
    data = pd.DataFrame(data={"income": [10, 20, 30], "year": [20, 30, 50], "height": [176, 195, 186]})

    # When
    numerical_columns = NumericalColumns.from_dataframe(data, columns_to_keep)

    # Then
    assert len(numerical_columns.columns) == 2
    assert numerical_columns.columns["income"].name == "income"
    assert numerical_columns.columns["year"].name == "year"
    assert numerical_columns.numerical_dimensions == 2


def test_build_numerical_columns_from_columns():
    # Given
    data = pd.DataFrame(data={"income": [10, 20, 30], "year": [20, 30, 50], "height": [176, 195, 186]})

    income_column = NumericalColumn.from_series(data["income"])
    height_column = NumericalColumn.from_series(data["height"])

    # When
    numerical_columns = NumericalColumns.from_columns([income_column, height_column])

    # Then
    assert len(numerical_columns.columns) == 2
    assert numerical_columns.columns["income"].name == "income"
    assert numerical_columns.columns["height"].name == "height"
    assert numerical_columns.numerical_dimensions == 2
