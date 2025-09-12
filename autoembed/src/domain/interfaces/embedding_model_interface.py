from typing import Dict, List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from autoembed.src.domain.models.data.preprocessed_data import PreprocessedData, PreprocessedTextData
from autoembed.src.domain.models.data.dataset_analysis import DatasetAnalysis


class EmbeddingModelInterface(ABC):
    @abstractmethod
    def fit(
        self,
        x: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int,
    ) -> None:
        pass

    @abstractmethod
    def from_dataset_analysis(
        self,
        dataset_analysis: DatasetAnalysis,
        bottleneck_layer_dim: int,
        hidden_layer_dim: List[int],
    ) -> "EmbeddingModelInterface":
        pass

    @abstractmethod
    def embed(self, x: PreprocessedData) -> np.ndarray:
        pass

    @abstractmethod
    def embed_text_column(self, x: PreprocessedTextData | PreprocessedData) -> np.ndarray:
        pass

    @abstractmethod
    def load(self, path: str) -> "EmbeddingModelInterface":
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass
