from abc import ABC, abstractmethod


class ClassifiedVectorsRepositoryInterface(ABC):
    @abstractmethod
    def get_most_similar_classifieds_by_classified_ref(self, classified_ref: str, limit: int) -> list[str]:
        pass

    @abstractmethod
    def get_most_similar_classifieds_by_vector(self, vector: list[float], limit: int) -> list[str]:
        pass

    @abstractmethod
    def update_classified_vector(self, classified_ref: str, vector: list[float]) -> None:
        pass

    @abstractmethod
    def update_batch_classified_vectors(self, classified_vectors: dict[str, list[float]]) -> None:
        pass
