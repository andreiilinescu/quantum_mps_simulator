import utils
import plotting
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from mps import MPS
from db_contraction import abstractDB,duckDB,sqliteDB
from simulator import Simulator
MAX_BOND=10
class MpsSimulator(Simulator):
    def __init__(self, num_qbits: int, db:abstractDB|None=None):
        self.num_qbits = num_qbits
        db_contraction=None
        if db is not None:
            self.db:abstractDB=db()
            db_contraction=self.db.contraction
        self.sim= MPS(num_qbits,max_bond=MAX_BOND ,db_contraction=db_contraction)
        self.qiskit_circ = QuantumCircuit(num_qbits)
        self.gates=[]

    def h(self, qbit: int):
        self.sim.apply_one_qubit_gate(qbit,utils.H())
        self.qiskit_circ.h(qbit)
        self.gates.append(('h',qbit))

    def x(self, qbit: int):
        self.sim.apply_one_qubit_gate(qbit,utils.X())
        self.qiskit_circ.x(qbit)
        self.gates.append(('x',qbit))


    def y(self, qbit: int):
        self.sim.apply_one_qubit_gate(qbit,utils.Y())
        self.qiskit_circ.y(qbit)
        self.gates.append(('y',qbit))


    def z(self, qbit: int):
        self.sim.apply_one_qubit_gate(qbit,utils.Z())
        self.qiskit_circ.z(qbit)
        self.gates.append(('z',qbit))


    def s(self, qbit: int):
        self.sim.apply_one_qubit_gate(qbit,utils.S())
        self.qiskit_circ.s(qbit)
        self.gates.append(('s',qbit))


    def sdg(self, qbit: int):
        self.sim.apply_one_qubit_gate(qbit,utils.SDG())
        self.qiskit_circ.sdg(qbit)
        self.gates.append(('sdg',qbit))


    def t(self, qbit: int):
        self.sim.apply_one_qubit_gate(qbit,utils.T())
        self.qiskit_circ.t(qbit)
        self.gates.append(('t',qbit))


    def p(self, qbit: int, angle: float):
        """
        Args:
            qbit (int): The number of the qbit we are applying the gate to
            angle (float): The phase shift angle in radians
        """
        self.sim.apply_one_qubit_gate(qbit,utils.P(angle))
        self.qiskit_circ.p(angle, qbit)
        self.gates.append(('p',qbit, angle))


    def cnot(self, control: int, target: int):
        self.sim.apply_two_qubit(control,target,utils.CNOT())
        self.qiskit_circ.cx(control, target)
        self.gates.append(('cnot',control, target))

    def ch(self, control: int, target: int):
        self.sim.apply_two_qubit(control,target,utils.CH())
        self.qiskit_circ.ch(control, target)
        self.gates.append(('ch',control, target))

    def cx(self, control: int, target: int):
        self.sim.apply_two_qubit(control,target,utils.CX())
        self.qiskit_circ.cx(control, target)
        self.gates.append(('cx',control, target))

    def cy(self, control: int, target: int):
        self.sim.apply_two_qubit(control,target,utils.CY())
        self.qiskit_circ.cy(control, target)
        self.gates.append(('cy',control, target))


    def cz(self, control: int, target: int):
        self.sim.apply_two_qubit(control,target,utils.CZ())
        self.qiskit_circ.cz(control, target)
        self.gates.append(('cz',control, target))

    def cs(self, control: int, target: int):
        self.sim.apply_two_qubit(control,target,utils.CS())
        self.qiskit_circ.cs(control, target)
        self.gates.append(('cs',control, target))

    def ct(self, control: int, target: int):
        self.sim.apply_two_qubit(control,target,utils.CT())
        self.qiskit_circ.ct(control, target)
        self.gates.append(('ct',control, target))

    def cp(self, control: int, target: int, angle: float):
        self.sim.apply_two_qubit(control,target,utils.CP(angle))
        self.qiskit_circ.cp(angle, control, target)
        self.gates.append(('cp',control, target, angle))
    
    def swap(self,qubit1:int,qubit2:int):
        self.sim.apply_two_qubit(qubit1,qubit2,utils.SWAP())
        self.qiskit_circ.swap(qubit1,qubit2)
        self.gates.append(('swap',qubit1, qubit2))
    
    # def toffoli(self, control1:int,control2:int,target:int):
    #     self.state=utils.apply_three_qubit_gate(self.state,utils.TOFFOLI(),control1,control2,target)
    #     self.qiskit_circ.ccx(control1,control2,target)
    #     self.gates.append(('toffoli',control1,control2, target))

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
        state=self.sim.get_statevector()
        out = np.array(state)
        out = np.around(np.abs(out.flatten()).__pow__(2), 5)
        return out

    def plot_qiskit_probabilities(self):
        probs = self.measure_qiskit_probablities()
        n = self.num_qbits
        labels = [f"{i:0{n}b}" for i in range(2**n)]
        plt.bar(labels, probs, color="skyblue")
        plt.xlabel("Quantum States")
        plt.ylabel("Probability")
        plt.title("Probability Distribution of Qiskit Quantum States")
        plt.ylim(0, 1)
        plt.show()

    def plot_own_probabilities(self):
        state=self.sim.get_statevector()
        plotting.plot_statevector(state)

    def draw_qiskit_circuit(self):
        print(self.qiskit_circ.draw(output='text'))

    def compare_results(self, print_res:bool=True):
        own = self.measure_own_probabilities()
        qiskit = self.measure_qiskit_probablities()
        same=np.array_equal(own, qiskit)
        result = (
            colored("same", "green")
            if same
            else colored("different", "red")
        )
        if print_res:
            print("Own:")
            print(own)
            print("Qiskit:")
            print(qiskit)
            print(f"Conclusion: {result}")

        return same

    # def compare_own_times(self,justDiff=False,last=False):
    #     if not justDiff:
    #         print("NumPY times:")
    #         print(self.sim_np.times)
    #         print("SQL times:")
    #         print(self.sim_sql.times)
    #     n=sum(self.sim_np.times)
    #     s=sum(self.sim_sql.times)
    #     if s>n:
    #         print("Diff:"+colored(s-n,"red"))
    #     else:
    #         print("Diff:"+colored(s-n,"green"))
    #     if last:
    #         print("LAST DIFF:"+str(self.sim_np.times[-1]-self.sim_sql.times[-1]))
    
    def get_times(self):
        return np.array(self.sim.times)

    @staticmethod
    def run(num_qubits,gates,db=None):
        sim=MpsSimulator(num_qubits,db=db)
        for x in gates:
            getattr(sim,x[0])(*x[1:])
        if db:
            sim.db.close()
        return sim
    
    
if __name__ == "__main__":
    # # Create a Bell state with 2 qubits
    simulator = MpsSimulator(3,duckDB)
    simulator.h(1)  # Apply Hadamard gate to the first qubit
    simulator.cnot(
                1, 2
            )  # Apply CNOT gate with qubit 0 as control and qubit 1 as target

    # Remaining qubits (if any) are left in the |0⟩ state
    simulator.compare_results()
    simulator.plot_own_probabilities()
    