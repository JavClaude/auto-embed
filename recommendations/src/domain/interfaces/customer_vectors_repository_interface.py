from abc import ABC, abstractmethod


class CustomerVectorsRepositoryInterface(ABC):
    @abstractmethod
    def get_customer_vector(self, customer_ref: str) -> list[float]:
        pass

    @abstractmethod
    def update_customer_vector(self, customer_ref: str, vector: list[float]) -> None:
        pass

    @abstractmethod
    def get_batch_customer_vectors(self, customer_refs: list[str]) -> dict[str, list[float]]:
        pass

    @abstractmethod
    def update_batch_customer_vectors(self, customer_vectors: dict[str, list[float]]) -> None:
        pass
