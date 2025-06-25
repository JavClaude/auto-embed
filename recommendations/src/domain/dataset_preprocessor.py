import pandas as pd
from typing import Dict, List

from recommendations.src.domain.entites.columns import (
    CategoricalColumns,
    NumericalColumns,
)
from recommendations.src.domain.entites.dataset_analysis import DatasetAnalysis


NUMERICAL_INPUTS_FEATURES_KEY = "numerical_inputs_features"
NUMERICAL_OUTPUTS_KEY = "numerical_outputs"

NUMERICAL_COLUMNS_NAMES = [
    "vehicle_mileage",
    "vehicle_year",
    "vehicle_doors",
    "vehicle_trunk_volume",
    "vehicle_refined_quotation",
    "vehicle_power_din",
    "vehicle_rated_horse_power",
    "vehicle_max_power",
    "vehicle_consumption",
    "vehicle_co2",
    "constructor_warranty_duration",
    "price",
    "vehicle_weight",
    "vehicle_cubic",
    "vehicle_length",
    "vehicle_height",
    "vehicle_width",
    "initial_price",
    "vehicle_price_new",
]

CATEGORICAL_COLUMNS_NAMES = [
    "customer_type",
    "vehicle_seats",
    "zip_code",
    "vehicle_category",
    "vehicle_make",
    "vehicle_model",
    "vehicle_version",
    "vehicle_gearbox",
    "vehicle_energy",
    "vehicle_origin",
    "vehicle_external_color",
    "vehicle_internal_color",
    "vehicle_four_wheel_drive",
    "vehicle_pollution_norm",
    "vehicle_condition",
    "vehicle_motorization",
    "vehicle_commercial_name",
]


class DatasetPreprocessor:
    def __init__(
        self,
        numerical_columns_names: List[str],
        categorical_columns_names: List[str],
        numerical_columns: NumericalColumns | None = None,
        categorical_columns: CategoricalColumns | None = None,
    ):
        self.numerical_columns_names = numerical_columns_names
        self.categorical_columns_names = categorical_columns_names
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    def fit(self, dataframe: pd.DataFrame) -> None:
        print("Fitting dataset preprocessor")
        self.numerical_columns = NumericalColumns.from_dataframe(dataframe, columns=self.numerical_columns_names)
        self.categorical_columns = CategoricalColumns.from_dataframe(dataframe, columns=self.categorical_columns_names)

    def get_analysis(self) -> DatasetAnalysis:
        return DatasetAnalysis(self.numerical_columns, self.categorical_columns)

    def preprocess(self, dataframe: pd.DataFrame) -> Dict[str, pd.Series]:
        transformed_data = dataframe[self.numerical_columns_names + self.categorical_columns_names].copy()

        for column in self.numerical_columns.columns:
            transformed_data[column] = self.numerical_columns.columns[column].transform(transformed_data[column])

        for column in self.categorical_columns.columns:
            transformed_data[column] = self.categorical_columns.columns[column].transform(transformed_data[column])

        return {
            NUMERICAL_INPUTS_FEATURES_KEY: transformed_data[self.numerical_columns_names].values,
            **{feature_name: transformed_data[feature_name].values for feature_name in self.categorical_columns_names},
        }

    def preprocess_target(self, dataframe: pd.DataFrame) -> Dict[str, pd.Series]:
        transformed_data = dataframe[self.numerical_columns_names + self.categorical_columns_names].copy()

        for column in self.numerical_columns.columns:
            transformed_data[column] = self.numerical_columns.columns[column].transform(transformed_data[column])

        for column in self.categorical_columns.columns:
            transformed_data[column] = self.categorical_columns.columns[column].transform(transformed_data[column])

        return {
            NUMERICAL_OUTPUTS_KEY: transformed_data[self.numerical_columns_names].values,
            **{feature_name + "_outputs": transformed_data[feature_name].values for feature_name in self.categorical_columns_names},
        }

    @classmethod
    def from_columns(
        cls,
        numerical_columns: NumericalColumns,
        categorical_columns: CategoricalColumns,
    ) -> "DatasetPreprocessor":
        numerical_columns_names = [column for column in numerical_columns.columns.keys()]
        categorical_columns_names = [column for column in categorical_columns.columns.keys()]
        return cls(
            numerical_columns_names,
            categorical_columns_names,
            numerical_columns,
            categorical_columns,
        )
