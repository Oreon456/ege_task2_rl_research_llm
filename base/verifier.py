from abc import ABC, abstractmethod

class Verifier(ABC):

    @abstractmethod
    def verify(self, data, test_answer: str) -> bool:
        pass

    @abstractmethod
    def extract_answer(self, test_solution: str) -> str:
        pass
