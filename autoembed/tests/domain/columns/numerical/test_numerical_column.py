import pandas as pd
from autoembed.src.domain.columns.numerical.numerical_column import NumericalColumn


def test_build_numerical_column_series():
    # Given
    income_serie = pd.Series(data=[10, 20, 30], name="income")

    # When
    income_column = NumericalColumn.from_series(income_serie)

    # Then
    assert income_column.name == "income"
    assert income_column.mean == 20
    assert income_column.value_used_to_fill_na == 20
    assert income_column.std == 10


def test_when_i_want_to_apply_numerical_transformation_then_the_transformed_serie_is_returned():
    # Given
    income_serie = pd.Series(data=[10, 20, 30], name="income")

    serie_to_transformed = pd.Series(data=[8, 22, 29, pd.NA], name="income")

    income_column = NumericalColumn.from_series(income_serie)

    # When
    transformed_income_serie = income_column.transform(serie_to_transformed)

    # Then
    assert transformed_income_serie.to_list() == [-1.2, 0.2, 0.9, 0.0]
