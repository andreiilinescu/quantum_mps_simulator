from abc import ABC, abstractmethod
import numpy as np
class Simulator(ABC):
    @staticmethod
    @abstractmethod
    def run(num_qubits,gates,db_contraction):
        pass
    @abstractmethod
    def get_times(self):
        pass