from dataclasses import dataclass
from typing import List


@dataclass
class TrainClassifiedEmbeddingsModelCommand:
    online_date: str
    bottle_neck_size: int
    hidden_layer_sizes: List[int]
    epochs: int
    batch_size: int
    light_mode: bool
