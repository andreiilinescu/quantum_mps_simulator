import nptRepr as npt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from MPS import MPS

MAX_BOND=10
class SimMPS:
    def __init__(self, qbits: int):
        self.qbits = qbits
        self.sim_np = MPS(qbits,max_bond=MAX_BOND,use_sql=False)
        self.sim_sql= MPS(qbits,max_bond=MAX_BOND,use_sql=True)
        self.qiskit_circ = QuantumCircuit(qbits)
        self.gates=[]

    def h(self, qbit: int):
        self.sim_np.apply_one_qubit_gate(qbit,npt.H())
        self.sim_sql.apply_one_qubit_gate(qbit,npt.H())
        self.qiskit_circ.h(qbit)
        self.gates.append(('h',qbit))

    def x(self, qbit: int):
        self.sim_np.apply_one_qubit_gate(qbit,npt.X())
        self.sim_sql.apply_one_qubit_gate(qbit,npt.X())
        self.qiskit_circ.x(qbit)
        self.gates.append(('x',qbit))


    def y(self, qbit: int):
        self.sim_np.apply_one_qubit_gate(qbit,npt.Y())
        self.sim_sql.apply_one_qubit_gate(qbit,npt.Y())
        self.qiskit_circ.y(qbit)
        self.gates.append(('y',qbit))


    def z(self, qbit: int):
        self.sim_np.apply_one_qubit_gate(qbit,npt.Z())
        self.sim_sql.apply_one_qubit_gate(qbit,npt.Z())
        self.qiskit_circ.z(qbit)
        self.gates.append(('z',qbit))


    def s(self, qbit: int):
        self.sim_np.apply_one_qubit_gate(qbit,npt.S())
        self.sim_sql.apply_one_qubit_gate(qbit,npt.S())
        self.qiskit_circ.s(qbit)
        self.gates.append(('s',qbit))


    def sdg(self, qbit: int):
        self.sim_np.apply_one_qubit_gate(qbit,npt.SDG())
        self.sim_sql.apply_one_qubit_gate(qbit,npt.SDG())
        self.qiskit_circ.sdg(qbit)
        self.gates.append(('sdg',qbit))


    def t(self, qbit: int):
        self.sim_np.apply_one_qubit_gate(qbit,npt.T())
        self.sim_sql.apply_one_qubit_gate(qbit,npt.T())
        self.qiskit_circ.t(qbit)
        self.gates.append(('t',qbit))


    def p(self, qbit: int, angle: float):
        """
        Args:
            qbit (int): The number of the qbit we are applying the gate to
            angle (float): The phase shift angle in radians
        """
        self.sim_np.apply_one_qubit_gate(qbit,npt.P(angle))
        self.sim_sql.apply_one_qubit_gate(qbit,npt.P(angle))
        self.qiskit_circ.p(angle, qbit)
        self.gates.append(('p',qbit, angle))


    def cnot(self, control: int, target: int):
        self.sim_np.apply_two_qubit(control,target,npt.CNOT())
        self.sim_sql.apply_two_qubit(control,target,npt.CNOT())
        self.qiskit_circ.cx(control, target)
        self.gates.append(('cnot',control, target))

    def ch(self, control: int, target: int):
        self.sim_np.apply_two_qubit(control,target,npt.CH())
        self.sim_sql.apply_two_qubit(control,target,npt.CH())
        self.qiskit_circ.ch(control, target)
        self.gates.append(('ch',control, target))

    def cx(self, control: int, target: int):
        self.sim_np.apply_two_qubit(control,target,npt.CX())
        self.sim_sql.apply_two_qubit(control,target,npt.CX())
        self.qiskit_circ.cx(control, target)
        self.gates.append(('cx',control, target))

    def cy(self, control: int, target: int):
        self.sim_np.apply_two_qubit(control,target,npt.CY())
        self.sim_sql.apply_two_qubit(control,target,npt.CY())
        self.qiskit_circ.cy(control, target)
        self.gates.append(('cy',control, target))


    def cz(self, control: int, target: int):
        self.sim_np.apply_two_qubit(control,target,npt.CZ())
        self.sim_sql.apply_two_qubit(control,target,npt.CZ())
        self.qiskit_circ.cz(control, target)
        self.gates.append(('cz',control, target))

    def cs(self, control: int, target: int):
        self.sim_np.apply_two_qubit(control,target,npt.CS())
        self.sim_sql.apply_two_qubit(control,target,npt.CS())
        self.qiskit_circ.cs(control, target)
        self.gates.append(('cs',control, target))

    def ct(self, control: int, target: int):
        self.sim_np.apply_two_qubit(control,target,npt.CT())
        self.sim_sql.apply_two_qubit(control,target,npt.CT())
        self.qiskit_circ.ct(control, target)
        self.gates.append(('ct',control, target))

    def cp(self, control: int, target: int, angle: float):
        self.sim_np.apply_two_qubit(control,target,npt.CP(angle))
        self.sim_sql.apply_two_qubit(control,target,npt.CP(angle))
        self.qiskit_circ.cp(angle, control, target)
        self.gates.append(('cp',control, target, angle))
    
    def swap(self,qubit1:int,qubit2:int):
        self.sim_np.apply_two_qubit(qubit1,qubit2,npt.SWAP())
        self.sim_sql.apply_two_qubit(qubit1,qubit2,npt.SWAP())
        self.qiskit_circ.swap(qubit1,qubit2)
        self.gates.append(('swap',qubit1, qubit2))
    
    def toffoli(self, control1:int,control2:int,target:int):
        self.state=npt.apply_three_qubit_gate(self.state,npt.TOFFOLI(),control1,control2,target)
        self.qiskit_circ.ccx(control1,control2,target)
        self.gates.append(('toffoli',control1,control2, target))

    def get_qiskit_statevector(self) -> Statevector:
        circ = self.qiskit_circ.copy().reverse_bits()
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
        state=self.sim_sql.get_statevector()
        out = np.array(state)
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
        state=self.sim_sql.get_statevector()
        npt.plot_nqbit_prob(state)

    def draw_qiskit_circuit(self):
        print(self.qiskit_circ.draw(output='text'))

    def compare_results(self, print:bool=True):
        own = self.measure_own_probabilities()
        qiskit = self.measure_qiskit_probablities()
        same=np.array_equal(own, qiskit)
        result = (
            colored("same", "green")
            if same
            else colored("different", "red")
        )
        if print:
            print("Own:")
            print(own)
            print("Qiskit:")
            print(qiskit)
            print(f"Conclusion: {result}")

        return same

    def compare_own_times(self,justDiff=False,last=False):
        if not justDiff:
            print("NumPY times:")
            print(self.sim_np.times)
            print("SQL times:")
            print(self.sim_sql.times)
        n=sum(self.sim_np.times)
        s=sum(self.sim_sql.times)
        if s>n:
            print("Diff:"+colored(s-n,"red"))
        else:
            print("Diff:"+colored(s-n,"green"))
        if last:
            print("LAST DIFF:"+str(self.sim_np.times[-1]-self.sim_sql.times[-1]))
    
    def get_times(self):
        return {'np':{
                    'times':self.sim_np.times,
                    'total':sum(self.sim_np.times)
                },
                'sql':{
                    'times':self.sim_sql.times,
                    'total':sum(self.sim_sql.times)
                }}

    @staticmethod
    def run(num_qubits,gates):
        sim=SimMPS(num_qubits)
        for x in gates:
            getattr(sim,x[0])(*x[1:])
        
        return sim
            

    @classmethod
    def setup_bell_state(cls, num_qubits: int) -> "SimMPS":
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
    def setup_ghz_state(cls, num_qubits: int) -> "SimMPS":
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
    def setup_phase_shift_example_state(cls) -> "SimMPS":
        simulator = cls(1)
        simulator.h(0)
        simulator.p(0, np.pi / 4)
        simulator.h(0)
        return simulator
    
    
    
if __name__ == "__main__":
    # # Create a Bell state with 2 qubits
    bell_simulator = SimMPS(3)
    bell_simulator.h(0)
    bell_simulator.h(1)
    bell_simulator.swap(0,1)
    bell_simulator.swap(1,2)
    bell_simulator.draw_qiskit_circuit()
    x=SimMPS.run(3,bell_simulator.gates)
    # for i in range(10):
    #     bell_simulator.cy(0,1)
    # print(bell_simulator.sim_sql.tensors)
    x.draw_qiskit_circuit()
    # bell_simulator.compare_results()
    # bell_simulator.compare_own_times(justDiff=True,last=True)
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