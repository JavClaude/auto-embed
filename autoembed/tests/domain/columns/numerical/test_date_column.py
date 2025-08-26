import pandas as pd
from autoembed.src.domain.columns.numerical.date_column import DateColumn


def test_when_to_transform_date_column_then_the_column_is_transform():
    # Given
    data = pd.DataFrame({"age": [22, 23, 24, 26], "date": ["2018-03-12", "2013-04-15", "2013", ""]})
    date_column = DateColumn.from_series(data["date"])

    # When
    transformed_date = date_column.transform(data["date"])

    # Then
    assert transformed_date[0] == 119.0
    assert transformed_date[1] == 113.36602540378443
    assert transformed_date[2] == 114.36602540378443
    assert transformed_date[3] == 114.36602540378443
