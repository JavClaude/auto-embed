import numpy as np
import pandas as pd
from typing import Dict, List

from autoembed.src.domain.columns.categorical.text_column import TextColumn
from autoembed.src.domain.models.dataset_analysis import DatasetAnalysis
from autoembed.src.domain.columns.numerical.numerical_columns import NumericalColumns
from autoembed.src.domain.columns.categorical.categorical_columns import CategoricalColumns
from autoembed.src.domain.models.preprocessed_data import PreprocessedData
from autoembed.src.domain.models.preprocessed_target import PreprocessedTarget


NUMERICAL_INPUTS_FEATURES_KEY = "numerical_inputs_features"
NUMERICAL_OUTPUTS_KEY = "numerical_outputs"


class DatasetPreprocessor:
    def __init__(
        self,
        numerical_columns_names: List[str] | None = [],  # TODO: introduce a NumericalColumnSpec
        categorical_columns_names: List[str] | None = [],  # TODO: introduce a CategoricalColumnSpec
        text_column_name: str | None = None,  # TODO: introduce a TextualColumnSpec
        numerical_columns: NumericalColumns | None = None,
        categorical_columns: CategoricalColumns | None = None,
        text_column: TextColumn | None = None,
        categorical_features_loss_weights: Dict[str, float] | None = None,
    ):
        if not numerical_columns_names and not categorical_columns_names and not text_column_name and not numerical_columns and not categorical_columns and not text_column:
            raise ValueError("numerical_columns_names or categorical_columns_names or text_column_name or numerical_columns or categorical_columns or text_column must be provided")

        self.numerical_columns_names = numerical_columns_names
        self.categorical_columns_names = categorical_columns_names
        self.text_column_name = text_column_name
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.text_column = text_column
        self.categorical_features_loss_weights = categorical_features_loss_weights

    @classmethod
    def from_columns(
        cls,
        numerical_columns: NumericalColumns | None = None,
        categorical_columns: CategoricalColumns | None = None,
        text_column: TextColumn | None = None,
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

        if text_column:
            text_column_name = text_column.name
        else:
            text_column_name = None

        return cls(numerical_columns_names, categorical_columns_names, text_column_name, numerical_columns, categorical_columns, text_column, categorical_features_loss_weights)

    def fit(self, dataframe: pd.DataFrame) -> None:
        if self.numerical_columns_names:
            self.numerical_columns = NumericalColumns.from_dataframe(dataframe, columns=self.numerical_columns_names)

        if self.text_column_name:
            self.text_column = TextColumn.from_series(dataframe[self.text_column_name])

        if self.categorical_columns_names:
            self.categorical_columns = CategoricalColumns.from_dataframe(dataframe, columns=self.categorical_columns_names)
            self.categorical_features_loss_weights = self.compute_categorical_loss_weights(self.categorical_columns)

    def preprocess(self, dataframe: pd.DataFrame) -> PreprocessedData:
        if self.text_column_name:
            transformed_data = dataframe[self.numerical_columns_names + self.categorical_columns_names + [self.text_column_name]].copy()
        else:
            transformed_data = dataframe[self.numerical_columns_names + self.categorical_columns_names].copy()

        numerical_inputs_features = None
        categorical_inputs_features = None
        text_input_feature = None

        if self.numerical_columns:
            numerical_inputs_features = self.numerical_columns.transform(transformed_data[self.numerical_columns_names])

        if self.categorical_columns:
            categorical_inputs_features = {
                column_name: self.categorical_columns.columns[column_name].transform(transformed_data[column_name]) for column_name in self.categorical_columns.columns.keys()
            }

        if self.text_column:
            text_input_feature = self.text_column.transform(transformed_data[self.text_column_name])

        return PreprocessedData(
            numerical_inputs_features=numerical_inputs_features,
            categorical_inputs_features=categorical_inputs_features,
            text_input_feature=text_input_feature,
        )

    def preprocess_target(self, dataframe: pd.DataFrame) -> PreprocessedTarget:
        transformed_data = dataframe[self.numerical_columns_names + self.categorical_columns_names].copy()

        numerical_outputs = None
        categorical_outputs = None

        if self.numerical_columns:
            numerical_outputs = self.numerical_columns.transform(transformed_data[self.numerical_columns_names])

        if self.categorical_columns:
            categorical_outputs = {column_name: self.categorical_columns.columns[column_name].transform(transformed_data[column_name]) for column_name in self.categorical_columns.columns.keys()}

        return PreprocessedTarget(
            numerical_outputs=numerical_outputs,
            categorical_outputs=categorical_outputs,
        )

    def get_analysis(self) -> DatasetAnalysis:
        return DatasetAnalysis(self.numerical_columns, self.categorical_columns, self.text_column, self.categorical_features_loss_weights)

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
