from dataclasses import dataclass


@dataclass
class LossWeight:
    name: str
    weight: float

    @classmethod
    def from_dict(cls, name: str, weight: float) -> "LossWeight":
        return cls(name, weight)
