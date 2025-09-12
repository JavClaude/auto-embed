from typing import List
from dataclasses import dataclass


@dataclass
class MetadataColumns:
    columns: List[str] | None
