from typing import Dict
from dataclasses import dataclass

from recommendations.src.domain.entites.columns import (
    CategoricalColumns,
    NumericalColumns,
)


@dataclass
class DatasetAnalysis:
    numerical_columns: NumericalColumns
    categorical_columns: CategoricalColumns

    def get_analysis(self) -> Dict[str, float]:
        return {
            "numerical_columns_names": self.numerical_columns,
            "categorical_columns_names": self.categorical_columns,
        }
