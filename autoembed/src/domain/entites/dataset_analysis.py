from typing import Dict
from dataclasses import dataclass

from autoembed.src.domain.entites.columns import (
    CategoricalColumns,
    NumericalColumns,
)


@dataclass
class DatasetAnalysis:
    numerical_columns: NumericalColumns | None 
    categorical_columns: CategoricalColumns | None
    categorical_features_loss_weights: None | Dict[str, float] = None

    def get_analysis(self) -> Dict[str, float]:
        return {
            "numerical_columns_names": self.numerical_columns,
            "categorical_columns_names": self.categorical_columns,
            "categorical_features_loss_weights": self.categorical_features_loss_weights,
            }
