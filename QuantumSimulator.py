import nptRepr as npt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from MPS import MPS

class QuantumSimulator:
    def __init__(self, qbits: int):
        self.qbits = qbits
        self.state = npt.init_state(qbits)
        self.qiskit_circ = QuantumCircuit(qbits)

    def h(self, qbit: int):
        self.state = npt.apply_one_qubit_gate(self.state, npt.H(), qbit)
        self.qiskit_circ.h(qbit)

    def x(self, qbit: int):
        self.state = npt.apply_one_qubit_gate(self.state, npt.X(), qbit)
        self.qiskit_circ.x(qbit)

    def y(self, qbit: int):
        self.state = npt.apply_one_qubit_gate(self.state, npt.Y(), qbit)
        self.qiskit_circ.y(qbit)

    def z(self, qbit: int):
        self.state = npt.apply_one_qubit_gate(self.state, npt.Z(), qbit)
        self.qiskit_circ.z(qbit)

    def s(self, qbit: int):
        self.state = npt.apply_one_qubit_gate(self.state, npt.S(), qbit)
        self.qiskit_circ.s(qbit)

    def sdg(self, qbit: int):
        self.state = npt.apply_one_qubit_gate(self.state, npt.SDG(), qbit)
        self.qiskit_circ.sdg(qbit)

    def t(self, qbit: int):
        self.state = npt.apply_one_qubit_gate(self.state, npt.T(), qbit)
        self.qiskit_circ.t(qbit)

    def p(self, qbit: int, angle: float):
        """
        Args:
            qbit (int): The number of the qbit we are applying the gate to
            angle (float): The phase shift angle in radians
        """
        self.state = npt.apply_one_qubit_gate(self.state, npt.P(angle), qbit)
        self.qiskit_circ.p(angle, qbit)

    def cnot(self, control: int, target: int):
        print(self.state)
        self.state = npt.apply_two_qubit_gate(self.state, npt.CNOT(), control, target)
        self.qiskit_circ.cx(control, target)

    def ch(self, control: int, target: int):
        self.state = npt.apply_two_qubit_gate(self.state, npt.CH(), control, target)
        self.qiskit_circ.ch(control, target)

    def cx(self, control: int, target: int):
        self.state = npt.apply_two_qubit_gate(self.state, npt.CX(), control, target)
        self.qiskit_circ.cx(control, target)

    def cy(self, control: int, target: int):
        self.state = npt.apply_two_qubit_gate(self.state, npt.CY(), control, target)
        self.qiskit_circ.cy(control, target)

    def cz(self, control: int, target: int):
        self.state = npt.apply_two_qubit_gate(self.state, npt.CZ(), control, target)
        self.qiskit_circ.cz(control, target)

    def cs(self, control: int, target: int):
        self.state = npt.apply_two_qubit_gate(self.state, npt.CS(), control, target)
        self.qiskit_circ.cs(control, target)

    def ct(self, control: int, target: int):
        self.state = npt.apply_two_qubit_gate(self.state, npt.CT(), control, target)
        self.qiskit_circ.ct(control, target)

    def cp(self, control: int, target: int, angle: float):
        self.state = npt.apply_two_qubit_gate(self.state, npt.CP(angle), control, target)
        self.qiskit_circ.cp(angle, control, target)
    
    def swap(self,qubit1:int,qubit2:int):
        self.state=npt.apply_two_qubit_gate(self.state,npt.SWAP(),qubit1,qubit2)
        self.qiskit_circ.swap(qubit1,qubit2)
    
    def toffoli(self, control1:int,control2:int,target:int):
        self.state=npt.apply_three_qubit_gate(self.state,npt.TOFFOLI(),control1,control2,target)
        self.qiskit_circ.ccx(control1,control2,target)

    def get_qiskit_statevector(self) -> Statevector:
        circ = self.qiskit_circ.copy()
        circ.save_statevector()
        # Transpile for simulator
        simulator = AerSimulator(method="statevector")
        circ = transpile(circ, simulator)
        # Run and get statevector
        result = simulator.run(circ).result()
        statevector = result.get_statevector(circ)
        return statevector

    def measure_qiskit_probablities(self) -> np.array:
        statevector = self.get_qiskit_statevector()
        probs = statevector.probabilities(decimals=5)
        return np.array(probs)

    def measure_own_probabilities(self) -> np.array:
        out = np.array(self.state)
        out = np.around(np.abs(out.flatten()).__pow__(2), 5)
        return out

    def plot_qiskit_probabilities(self):
        probs = self.measure_qiskit_probablities()
        n = self.qbits
        labels = [f"{i:0{n}b}" for i in range(2**n)]
        plt.bar(labels, probs, color="skyblue")
        plt.xlabel("Quantum States")
        plt.ylabel("Probability")
        plt.title("Probability Distribution of Qiskit Quantum States")
        plt.ylim(0, 1)
        plt.show()

    def plot_own_probabilities(self):
        npt.plot_nqbit_prob(self.state)

    def draw_qiskit_circuit(self):
        print(self.qiskit_circ.draw())

    def compare_results(self):
        own = self.measure_own_probabilities()
        qiskit = self.measure_qiskit_probablities()
        result = (
            colored("same", "green")
            if np.array_equal(own, qiskit)
            else colored("different", "red")
        )

        print("Own:")
        print(own)
        print("Qiskit:")
        print(qiskit)
        print(f"Conclusion: {result}")
            

    @classmethod
    def setup_bell_state(cls, num_qubits: int) -> "QuantumSimulator":
        """
        Sets up a Bell state with the specified number of qubits.

        Args:
            num_qubits (int): The number of qubits in the GHZ state.

        Returns:
            QuantumSimulator: An instance of QuantumSimulator with GHZ state prepared.
        """
        if num_qubits < 2:
            raise ValueError(
                "Number of qubits must be at least 2 to create a Bell state."
            )

        simulator = cls(num_qubits)
        simulator.h(0)  # Apply Hadamard gate to the first qubit
        simulator.cnot(
            0, 1
        )  # Apply CNOT gate with qubit 0 as control and qubit 1 as target

        # Remaining qubits (if any) are left in the |0⟩ state
        return simulator

    @classmethod
    def setup_ghz_state(cls, num_qubits: int) -> "QuantumSimulator":
        """
        Sets up a GHZ state with the specified number of qubits.

        Args:
            num_qubits (int): The number of qubits in the GHZ state.

        Returns:
            QuantumSimulator: An instance of QuantumSimulator with GHZ state prepared.
        """
        if num_qubits < 2:
            raise ValueError(
                "Number of qubits must be at least 2 to create a GHZ state."
            )

        simulator = cls(num_qubits)
        simulator.h(0)  # Apply Hadamard gate to the first qubit

        # Apply CNOT gates from the first qubit to all other qubits
        for target in range(1, num_qubits):
            simulator.cnot(0, target)

        return simulator

    @classmethod
    def setup_phase_shift_example_state(cls) -> "QuantumSimulator":
        simulator = cls(1)
        simulator.h(0)
        simulator.p(0, np.pi / 4)
        simulator.h(0)
        return simulator
    
    
if __name__ == "__main__":
    # # Create a Bell state with 2 qubits
    bell_simulator = QuantumSimulator(2)
    bell_simulator.h(1)
    bell_simulator.cnot(0,1)

    # print(bell_simulator.state)
    # bell_simulator.draw_qiskit_circuit()
    bell_simulator.compare_results()
    # bell_simulator.plot_own_probabilities()
    # # q=QuantumSimulator(5)
    # simulator = QuantumSimulator.setup_phase_shift_example_state()
    # simulator.draw_qiskit_circuit()
    # simulator.compare_results()
    # simulator.plot_own_probabilities()
    # n=4
    # simulator = QuantumSimulator(n)  # QFT on 3 qubits
    # for i in range(n):
    #     simulator.h(i)
    #     for j in range(i + 1, n):
    #         angle = np.pi / (2 ** (j - i))
    #         simulator.cp(i, j, angle)
    # for i in range(n // 2):
    #     simulator.swap(i, n - i - 1)

    
    # simulator.draw_qiskit_circuit()
    # simulator.compare_results()
    # simulator.plot_own_probabilities()
    # m=MPS(5)
    # print(m.tensors)
    # print(m.get_statevector())