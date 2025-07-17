import numpy as np
import pandas as pd
from typing import Dict, List

from autoembed.src.domain.entites.columns import (
    CategoricalColumns,
    NumericalColumns,
)
from autoembed.src.domain.entites.dataset_analysis import DatasetAnalysis


NUMERICAL_INPUTS_FEATURES_KEY = "numerical_inputs_features"
NUMERICAL_OUTPUTS_KEY = "numerical_outputs"


class DatasetPreprocessor:
    def __init__(
        self,
        numerical_columns_names: List[str],
        categorical_columns_names: List[str],
        numerical_columns: NumericalColumns | None = None,
        categorical_columns: CategoricalColumns | None = None,
        categorical_features_loss_weights: Dict[str, float] | None = None,
    ):
        self.numerical_columns_names = numerical_columns_names
        self.categorical_columns_names = categorical_columns_names
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.categorical_features_loss_weights = categorical_features_loss_weights

    def fit(self, dataframe: pd.DataFrame) -> None:
        self.numerical_columns = NumericalColumns.from_dataframe(dataframe, columns=self.numerical_columns_names)
        self.categorical_columns = CategoricalColumns.from_dataframe(dataframe, columns=self.categorical_columns_names)
        self.categorical_features_loss_weights = self.compute_categorical_loss_weights()

    def get_analysis(self) -> DatasetAnalysis:
        return DatasetAnalysis(self.numerical_columns, self.categorical_columns, self.categorical_features_loss_weights)

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

    def compute_categorical_loss_weights(self, max_weight_cap: float = 5.0) -> Dict[str, float]:
        if not self.categorical_columns.columns:
            return {}
        
        vocab_sizes = {
            name: len(col.vocabulary) 
            for name, col in self.categorical_columns.columns.items()
        }
        
        min_size = min(vocab_sizes.values())
        
        weights = {}
        for name, size in vocab_sizes.items():
            if min_size == 1:
                raw_weight = float(np.log(size + 1)) if size > 1 else 1.0
            else:
                raw_weight = float(np.log(size) / np.log(min_size))
            
            weights[name] = float(min(raw_weight, max_weight_cap))
        
        return weights

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
        categorical_features_loss_weights: Dict[str, float] | None = None,
    ) -> "DatasetPreprocessor":
        numerical_columns_names = [column for column in numerical_columns.columns.keys()]
        categorical_columns_names = [column for column in categorical_columns.columns.keys()]
        return cls(
            numerical_columns_names,
            categorical_columns_names,
            numerical_columns,
            categorical_columns,
            categorical_features_loss_weights,
        )
