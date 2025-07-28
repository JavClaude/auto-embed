

import pandas as pd

from autoembed.src.domain.columns.categorical.categorical_columns import CategoricalColumns


def test_build_categorical_columns_from_dataframe():
    # Given
    data = pd.DataFrame(
        data={
            "city": ["Paris", pd.NA, " lyon", "  MARSEILLE   "],
            "region": ["Ile-de-France", "Ile-de-France", pd.NA, "Auvergne-Rhône-Alpes"],
            "country": ["France", "France", pd.NA, "France"],
        }
    )

    # When
    categorical_columns = CategoricalColumns.from_dataframe(data, ["city", "region"])

    # Then
    assert len(categorical_columns.columns) == 2
    assert categorical_columns.columns["city"].name == "city"
    assert categorical_columns.columns["region"].name == "region"
