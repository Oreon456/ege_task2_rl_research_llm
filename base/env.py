from abc import ABC, abstractmethod

class Env(ABC):

    def __init__(self, name: str, verifier):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(self, num_of_questions=100, max_attempts=100, difficulty=1, **kwargs):
        pass

    def verify(self, data, test_solution: str):
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str):
        pass
