from abc import ABC, abstractmethod
import numpy as np
class Simulator(ABC):
    @staticmethod
    @abstractmethod
    def run(num_qubits:int,gates:np.ndarray,db) ->'Simulator':
        pass
    @abstractmethod
    def get_times(self) -> np.ndarray:
        pass