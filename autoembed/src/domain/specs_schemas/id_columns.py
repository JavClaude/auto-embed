from dataclasses import dataclass
from typing import List


@dataclass
class IdColumns:
    columns: List[str]

    def __post_init__(self):
        if self.columns == []:
            raise ValueError("Id columns cannot be empty, please provide at least one column")
