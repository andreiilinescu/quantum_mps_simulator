import nptRepr as npt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

class QuantumSimulator:
    def __init__(self,qbits:int):
        self.qbits=qbits
        self.state=npt.init_state(qbits)
        self.qiskit_circ=QuantumCircuit(qbits)

    def h(self,qbit:int):
        self.state=npt.apply_one_qubit_gate(self.state,npt.H(),qbit)
        self.qiskit_circ.h(qbit)

    def x(self,qbit:int):
        self.state=npt.apply_one_qubit_gate(self.state,npt.X(),qbit)
        self.qiskit_circ.x(qbit)
    
    def y(self,qbit:int):
        self.state=npt.apply_one_qubit_gate(self.state,npt.Y(),qbit)
        self.qiskit_circ.y(qbit)
    
    def z(self,qbit:int):
        self.state=npt.apply_one_qubit_gate(self.state,npt.Z(),qbit)
        self.qiskit_circ.z(qbit)


    def CNOT(self,control:int,target:int):
        self.state=npt.apply_two_qubit_gate(self.state,npt.CNOT(),control,target)
        self.qiskit_circ.cx(control,target)



    def get_qiskit_statevector(self) -> Statevector:
        circ=self.qiskit_circ.copy()
        circ.save_statevector()
        # Transpile for simulator
        simulator = AerSimulator(method='statevector')
        circ = transpile(circ, simulator)
        # Run and get statevector
        result = simulator.run(circ).result()
        statevector = result.get_statevector(circ)
        return statevector
    
    def measure_qiskit_probablities(self) -> np.array:
        statevector=self.get_qiskit_statevector()
        probs=statevector.probabilities(decimals=2)
        return np.array(probs)
    
    def measure_own_probabilities(self) -> np.array:
        out=np.array(self.state)
        out=np.around(np.abs(out.flatten()).__pow__(2),2)
        return out
    
    def plot_qiskit_probabilities(self):
        probs=self.measure_qiskit_probablities()
        n = self.qbits
        labels = [f'{i:0{n}b}' for i in range(2**n)]
        print(probs)
        plt.bar(labels,probs, color='skyblue')
        plt.xlabel('Quantum States')
        plt.ylabel('Probability')
        plt.title('Probability Distribution of Qiskit Quantum States')
        plt.ylim(0, 1)
        plt.show()
    
    def plot_own_probabilities(self):
        npt.plot_nqbit_prob(self.state)

    def draw_qiskit_circuit(self):
        print(self.qiskit_circ.draw())
    
    def compare_results(self):
        own=self.measure_own_probabilities()
        qiskit=self.measure_qiskit_probablities()
        result=colored("same","green") if np.array_equal(own,qiskit) else colored("different","red")
        
        print("Own:")
        print(own)
        print("Qiskit:")
        print(qiskit)
        print(f"Conclusion: {result}")

    


if __name__ == "__main__":
    q=QuantumSimulator(4)
    q.h(0)
    q.h(2)
    q.x(2)
    q.y(1)
    q.CNOT(0,1)
    q.CNOT(2,1)
    q.draw_qiskit_circuit()
    q.compare_results()
    q.plot_own_probabilities()
