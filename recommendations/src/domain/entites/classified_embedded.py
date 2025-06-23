from dataclasses import dataclass


@dataclass
class ClassifiedEmbedded:
    classified_ref: str
    vector: list[float]
