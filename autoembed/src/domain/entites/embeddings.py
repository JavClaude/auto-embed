import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class BusinessEmbeddings:
    def __init__(self, id_column_name: str, embeddings: np.ndarray, metadata: Dict[str, Any]):
        self.id_column_name = id_column_name
        self.embeddings = embeddings
        self.metadata = metadata

    def __str__(self) -> str:
        return f"Embedding(id={self.id_column_name}, embeddings={self.embeddings}, metadata={self.metadata})"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            **self.metadata,
        }


@dataclass
class BatchOfEmbeddings:
    embeddings: List[BusinessEmbeddings] = field(default_factory=list)

    def add_embeddings(self, embeddings: BusinessEmbeddings) -> None:
        self.embeddings.append(embeddings)

    def __len__(self) -> int:
        return len(self.embeddings)
    
    def sample_batch(self, n: int) -> "BatchOfEmbeddings":
        return BatchOfEmbeddings(random.sample(self.embeddings, n))
    