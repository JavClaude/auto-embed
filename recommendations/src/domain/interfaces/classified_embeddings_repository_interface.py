import random
from typing import Any, Dict, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class ClassifiedEmbeddings:
    classified_ref: str
    classified_embeddings: np.ndarray
    vehicle_make: str = "N/A"
    vehicle_model: str = "N/A"
    vehicle_commercial_name: str = "N/A"
    vehicle_version: str = "N/A"
    vehicle_energy: str = "N/A"
    vehicle_price: float = 0.0
    vehicle_year: int = 0


    def __str__(self) -> str:
        return f"ClassifiedEmbeddings(classified_ref={self.classified_ref}, vehicle_make={self.vehicle_make}, vehicle_model={self.vehicle_model}, vehicle_commercial_name={self.vehicle_commercial_name}, vehicle_version={self.vehicle_version}, vehicle_energy={self.vehicle_energy}, vehicle_price={self.vehicle_price}, vehicle_year={self.vehicle_year})"

    def get_classified_metadata(self) -> Dict[str, Any]:
        return {
            "vehicle_make": self.vehicle_make,
            "vehicle_model": self.vehicle_model,
            "vehicle_commercial_name": self.vehicle_commercial_name,
            "vehicle_version": self.vehicle_version,
            "vehicle_energy": self.vehicle_energy,
            "vehicle_price": self.vehicle_price,
            "vehicle_year": self.vehicle_year,
        }



@dataclass
class ClassifiedEmbeddingsBatch:
    classified_embeddings: List[ClassifiedEmbeddings] = field(default_factory=list)

    def add_classified_embeddings(self, classified_embeddings: ClassifiedEmbeddings) -> None:
        self.classified_embeddings.append(classified_embeddings)

    def __len__(self) -> int:
        return len(self.classified_embeddings)
    
    def sample_batch(self, n: int) -> "ClassifiedEmbeddingsBatch":
        return ClassifiedEmbeddingsBatch(random.sample(self.classified_embeddings, n))
    

class ClassifiedEmbeddingsRepositoryInterface(ABC):
    @abstractmethod
    def get_classified_embeddings(self, classified_ref: str) -> ClassifiedEmbeddings:
        pass

    @abstractmethod
    def get_most_similar_classified_embeddings(self, classified_ref: str, n: int = 10) -> List[ClassifiedEmbeddings]:
        pass

    @abstractmethod
    def update_classified_embeddings(self, classified_embeddings: ClassifiedEmbeddings) -> None:
        pass

    @abstractmethod
    def update_batch(self, classified_embeddings_batch: ClassifiedEmbeddingsBatch) -> None:
        pass

    @abstractmethod
    def get_classified_embeddings_batch(self, classified_refs: List[str]) -> List[ClassifiedEmbeddings]:
        pass

    @abstractmethod
    def get_all_embeddings(self) -> ClassifiedEmbeddingsBatch:
        pass
