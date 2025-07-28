import numpy as np
import pandas as pd
from typing import Dict, List

from autoembed.src.domain.models.dataset_analysis import DatasetAnalysis
from autoembed.src.domain.columns.numerical.numerical_columns import NumericalColumns
from autoembed.src.domain.columns.categorical.categorical_columns import CategoricalColumns


NUMERICAL_INPUTS_FEATURES_KEY = "numerical_inputs_features"
NUMERICAL_OUTPUTS_KEY = "numerical_outputs"


class DatasetPreprocessor:
    def __init__(
        self,
        numerical_columns_names: List[str] | None = [],
        categorical_columns_names: List[str] | None = [],
        numerical_columns: NumericalColumns | None = None,
        categorical_columns: CategoricalColumns | None = None,
        categorical_features_loss_weights: Dict[str, float] | None = None,
    ):
        if not numerical_columns_names and not categorical_columns_names and not numerical_columns and not categorical_columns:
            raise ValueError("numerical_columns_names or categorical_columns_names or numerical_columns or categorical_columns must be provided")

        self.numerical_columns_names = numerical_columns_names
        self.categorical_columns_names = categorical_columns_names
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.categorical_features_loss_weights = categorical_features_loss_weights

    @classmethod
    def from_columns(
        cls,
        numerical_columns: NumericalColumns | None = None,
        categorical_columns: CategoricalColumns | None = None,
        categorical_features_loss_weights: Dict[str, float] | None = None,
    ) -> "DatasetPreprocessor":
        if numerical_columns:
            numerical_columns_names = [column for column in numerical_columns.columns.keys()]
        else:
            numerical_columns_names = []

        if categorical_columns:
            categorical_columns_names = [column for column in categorical_columns.columns.keys()]
        else:
            categorical_columns_names = []

        return cls(
            numerical_columns_names,
            categorical_columns_names,
            numerical_columns,
            categorical_columns,
            categorical_features_loss_weights,
        )

    def fit(self, dataframe: pd.DataFrame) -> None:
        if self.numerical_columns_names:
            self.numerical_columns = NumericalColumns.from_dataframe(dataframe, columns=self.numerical_columns_names)

        if self.categorical_columns_names:
            self.categorical_columns = CategoricalColumns.from_dataframe(dataframe, columns=self.categorical_columns_names)
            self.categorical_features_loss_weights = self.compute_categorical_loss_weights(self.categorical_columns)

    def preprocess(self, dataframe: pd.DataFrame) -> Dict[str, np.array]:
        transformed_data = dataframe[self.numerical_columns_names + self.categorical_columns_names].copy()

        # TODO: Add real object instead of dict
        transformed_features = {}

        if self.numerical_columns:
            for column in self.numerical_columns.columns:
                transformed_data[column] = self.numerical_columns.columns[column].transform(transformed_data[column])

            transformed_features[NUMERICAL_INPUTS_FEATURES_KEY] = transformed_data[self.numerical_columns_names].values

        if self.categorical_columns:
            for column in self.categorical_columns.columns:
                transformed_features[column] = self.categorical_columns.columns[column].transform(transformed_data[column])

        return transformed_features

    def preprocess_target(self, dataframe: pd.DataFrame) -> Dict[str, pd.Series]:
        transformed_data = dataframe[self.numerical_columns_names + self.categorical_columns_names].copy()

        # TODO: Add real object instead of dict
        transformed_target = {}

        if self.numerical_columns:
            for column in self.numerical_columns.columns:
                transformed_data[column] = self.numerical_columns.columns[column].transform(transformed_data[column])

            transformed_target[NUMERICAL_OUTPUTS_KEY] = transformed_data[self.numerical_columns_names].values

        if self.categorical_columns:
            for column in self.categorical_columns.columns:
                transformed_target[column + "_outputs"] = self.categorical_columns.columns[column].transform(transformed_data[column])

        return transformed_target

    def get_analysis(self) -> DatasetAnalysis:
        return DatasetAnalysis(self.numerical_columns, self.categorical_columns, self.categorical_features_loss_weights)

    @staticmethod
    def compute_categorical_loss_weights(categorical_columns: CategoricalColumns | None, max_weight_cap: float = 5.0) -> Dict[str, float]:
        all_categorical_columns_loss_weight = {}

        if not categorical_columns.columns:
            return all_categorical_columns_loss_weight

        columns_vocabulary = {column_name: len(col.vocabulary) for column_name, col in categorical_columns.columns.items()}

        min_size = min(columns_vocabulary.values())

        for name, size in columns_vocabulary.items():

            if min_size == 1:
                raw_weight = float(np.log(size + 1)) if size > 1 else 1.0
            else:
                raw_weight = float(np.log(size) / np.log(min_size))

            weights = float(min(raw_weight, max_weight_cap))

            all_categorical_columns_loss_weight[name] = weights

        return all_categorical_columns_loss_weight
