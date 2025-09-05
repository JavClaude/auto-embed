import dataclasses
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from autoembed.src.domain.columns.categorical.categorical_column import CategoricalColumn
from autoembed.src.domain.columns.categorical.text_column import TextColumn
from autoembed.src.domain.columns.numerical.date_column import DateColumn
from autoembed.src.domain.columns.numerical.numerical_column import NumericalColumn
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
        date_columns_names: List[str] | None = [],
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
        self.date_columns_names = date_columns_names
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

        return DatasetPreprocessor(
            numerical_columns_names=numerical_columns_names,
            categorical_columns_names=categorical_columns_names,
            text_column_name=text_column_name,
            date_columns_names=None,  # We don't need to pass date columns names here because they are already in the numerical columns
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_column=text_column,
            categorical_features_loss_weights=categorical_features_loss_weights,
        )

    @classmethod
    def from_dict_definition(cls, data: Dict[str, Any]) -> "DatasetPreprocessor":
        numerical_columns = []
        for numerical_column in data["numerical_columns"]:
            categorical_column_attributes = list(numerical_column.values())[0]
            if categorical_column_attributes["is_date"]:
                column = DateColumn(**categorical_column_attributes)
            else:
                column = NumericalColumn(**categorical_column_attributes)

            numerical_columns.append(column)

        categorical_columns = []
        for categorical_column in data["categorical_columns"]:
            categorical_column_attributes = list(categorical_column.values())[0]
            column = CategoricalColumn(**categorical_column_attributes)
            categorical_columns.append(column)

        text_column = None
        if "text_column" in data:
            tokenizer = Tokenizer.from_file(data["text_column"]["tokenizer_path"])
            text_columns_attributes = data["text_column"]["text_columnns_attr"]
            text_column = TextColumn.from_tokenizer(
                tokenizer,
                text_columns_attributes["name"],
                text_columns_attributes["vocab_size"],
                text_columns_attributes["max_length"],
                text_columns_attributes["max_vocab_size"],
                text_columns_attributes["word_embedding"],
            )

        return DatasetPreprocessor(None, None, None, None, NumericalColumns.from_columns(numerical_columns), CategoricalColumns.from_categorical_columns(categorical_columns), text_column, None)

    def export_as_dict(self, path: str) -> Dict[str, Any]:
        preprocessor_data = {
            "numerical_columns": [{column_name: dataclasses.asdict(column)} for column_name, column in self.numerical_columns.columns.items()],
            "categorical_columns": [{column_name: dataclasses.asdict(column)} for column_name, column in self.categorical_columns.columns.items()],
            "categorical_features_loss_weights": self.categorical_features_loss_weights,
        }

        if self.text_column:
            tokenizer_path = f"{path}/tokenizer"

            preprocessor_data["text_column"] = {"text_columnns_attr": self.text_column.get_config(), "tokenizer_path": tokenizer_path}

        return preprocessor_data

    def has_a_text_column(self) -> bool:
        return self.text_column is not None

    def get_tokenizer(self) -> Tokenizer:
        if self.text_column:
            return self.text_column.tokenizer
        else:
            raise ValueError("No text column found")

    def fit(self, dataframe: pd.DataFrame) -> None:
        if self.numerical_columns_names:
            self.numerical_columns = NumericalColumns.from_dataframe(dataframe, numerical_columns_names=self.numerical_columns_names, date_columns_names=self.date_columns_names)

        if self.text_column_name:
            text_series = dataframe[self.text_column_name]
            self.text_column = TextColumn.from_series(text_series)

        if self.categorical_columns_names:
            self.categorical_columns = CategoricalColumns.from_dataframe(dataframe, columns=self.categorical_columns_names)
            self.categorical_features_loss_weights = self.compute_categorical_loss_weights(self.categorical_columns)

    def preprocess(self, dataframe: pd.DataFrame) -> PreprocessedData:
        if self.text_column:
            data_of_interest = dataframe[self.numerical_columns.get_all_columns_names() + self.categorical_columns.get_all_columns_names() + [self.text_column.name]].copy()
        else:
            data_of_interest = dataframe[self.numerical_columns.get_all_columns_names() + self.categorical_columns.get_all_columns_names()].copy()

        numerical_inputs_features = None
        categorical_inputs_features = None
        text_input_feature = None

        if self.numerical_columns:
            for column_name, column_processor in self.numerical_columns.columns.items():
                data_of_interest[column_name] = column_processor.transform(data_of_interest[column_name])

            numerical_inputs_features = data_of_interest[self.numerical_columns.get_all_columns_names()].values

        if self.categorical_columns:
            categorical_inputs_features = {
                f"{column_name}_inputs": self.categorical_columns.columns[column_name].transform(data_of_interest[column_name]) for column_name in self.categorical_columns.get_all_columns_names()
            }

        if self.text_column:
            text_input_feature = self.text_column.transform(data_of_interest[self.text_column.name])
            text_input_feature = {f"{self.text_column.name}_text_input": self.text_column.transform(data_of_interest[self.text_column.name]).input_ids}

        return PreprocessedData(
            numerical_inputs_features=numerical_inputs_features,
            categorical_inputs_features=categorical_inputs_features,
            text_input_feature=text_input_feature,
        )

    def preprocess_target(self, dataframe: pd.DataFrame) -> PreprocessedTarget:
        data_of_interest = dataframe[self.numerical_columns.get_all_columns_names() + self.categorical_columns.get_all_columns_names()].copy()

        numerical_outputs = None
        categorical_outputs = None

        if self.numerical_columns:
            for column_name, column_processor in self.numerical_columns.columns.items():
                data_of_interest[column_name] = column_processor.transform(data_of_interest[column_name])

            numerical_outputs = data_of_interest[self.numerical_columns.get_all_columns_names()].values

        if self.categorical_columns:
            categorical_outputs = {
                f"{column_name}_outputs": self.categorical_columns.columns[column_name].transform(data_of_interest[column_name]) for column_name in self.categorical_columns.columns.keys()
            }

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
