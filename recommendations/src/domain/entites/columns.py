import dataclasses
from typing import Dict, List
from abc import ABC, abstractmethod

import pandas as pd


class Column(ABC):
    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        pass


@dataclasses.dataclass
class CategoricalColumn(Column):
    name: str
    vocabulary: Dict[str, int]
    value_used_to_fill_na: str = "UNK"
    embedding_dim: int = 64

    def __post_init__(self):
        self.vocabulary = {
            **self.vocabulary,
            self.value_used_to_fill_na: len(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, name: str, vocabulary: Dict[str, int], embedding_dim: int) -> "CategoricalColumn":
        return cls(
            name=name,
            vocabulary=vocabulary,
            embedding_dim=embedding_dim,
        )

    @classmethod
    def from_series(cls, series: pd.Series) -> "CategoricalColumn":
        series.fillna(cls.value_used_to_fill_na, inplace=True)
        series = series.astype(str).apply(lambda x: x.lower().strip())
        vocabulary = {value: index for index, value in enumerate(series.unique())}
        return cls(
            name=series.name,
            vocabulary=vocabulary,
            embedding_dim=cls.infer_embedding_dim(vocabulary),
        )

    @staticmethod
    def infer_embedding_dim(vocabulary: Dict[str, int]) -> int:
        if len(vocabulary) < 30:
            return 20
        else:
            return 64

    def transform(self, series: pd.Series) -> pd.Series:
        series.fillna(self.value_used_to_fill_na, inplace=True)
        series = series.astype(str).apply(lambda x: x.lower().strip())
        return series.map(self.vocabulary)


@dataclasses.dataclass
class CategoricalColumns:
    columns: Dict[str, CategoricalColumn]

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, columns: List[str]) -> "CategoricalColumns":
        return cls(columns={column: CategoricalColumn.from_series(dataframe[column]) for column in columns})

    @classmethod
    def from_categorical_columns(cls, categorical_columns: List[CategoricalColumn]) -> "CategoricalColumns":
        return cls(columns={column.name: column for column in categorical_columns})


@dataclasses.dataclass
class NumericalColumn(Column):
    name: str
    value_used_to_fill_na: float | int
    mean: float
    std: float = 1.0

    @classmethod
    def from_dict(cls, name: str, mean: float, std: float) -> "NumericalColumn":
        return cls(name=name, value_used_to_fill_na=mean, mean=mean, std=std)

    @classmethod
    def from_series(cls, series: pd.Series) -> "NumericalColumn":
        mean = series.mean()
        series.fillna(mean, inplace=True)
        std = series.std()
        return NumericalColumn(name=series.name, value_used_to_fill_na=mean, mean=mean, std=std)

    def transform(self, series: pd.Series) -> pd.Series:
        series.fillna(self.value_used_to_fill_na, inplace=True)
        return (series - self.mean) / self.std


@dataclasses.dataclass
class NumericalColumns:
    columns: Dict[str, NumericalColumn]
    numerical_dimensions: int

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, columns: List[str]) -> "NumericalColumns":
        return cls(
            columns={column: NumericalColumn.from_series(dataframe[column]) for column in columns},
            numerical_dimensions=len(columns),
        )

    @classmethod
    def from_numerical_columns(cls, numerical_columns: List[NumericalColumn]) -> "NumericalColumns":
        return cls(
            columns={column.name: column for column in numerical_columns},
            numerical_dimensions=len(numerical_columns),
        )
