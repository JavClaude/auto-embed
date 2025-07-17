from dataclasses import dataclass


@dataclass
class UpdateUsersEmbeddingCommand:
    user_id: str
    classified_ref: str
    classified_event_type: str
