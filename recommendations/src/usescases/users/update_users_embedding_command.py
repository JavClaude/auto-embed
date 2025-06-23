from dataclasses import dataclass
from enum import Enum


@dataclass
class UpdateUsersEmbeddingCommand:
    user_id: str
    classified_ref: str
    classified_event_type: str
