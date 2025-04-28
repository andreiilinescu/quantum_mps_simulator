from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from timeit import default_timer as timer
import numpy as np
from performance import save_data_to_file

def prep_qft_qiskit_circ(n:int):
    circ = QuantumCircuit(n)
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            circ.cp(np.pi / (2 ** (j - i)), i, j)
    return circ

def prep_ghz_qiskit_circ(n:int):
    circ = QuantumCircuit(n)
    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    return circ

def run_circuit(circ):
    circ.save_matrix_product_state()
    sim = AerSimulator(method="matrix_product_state")
    result = sim.run(circ).result()
    return result.metadata

def simulate(circuit_creator, num_qubits_list:list, num_iters):
    full_times={}
    full_memory={}
    for num_qubits in num_qubits_list:
        times=[]
        memory=[]
        for _ in range(num_iters):
            circ = circuit_creator(num_qubits)
            meta=run_circuit(circ)
            times.append(meta["time_taken_execute"])
            memory.append(meta["max_memory_mb"])
        full_times[str(num_qubits)]=times
        full_memory[str(num_qubits)]=memory
    return full_times,full_memory

print(run_circuit(prep_qft_qiskit_circ(100)))

# MIN_QUBITS=200
# MAX_QUBITS=1000
# ITER=100
# STEP=100
# SYSTEM="PC-2080s"
# times,memory=simulate(prep_ghz_qiskit_circ, list(range(MIN_QUBITS,MAX_QUBITS+1,STEP)), ITER)
# for i in range(MIN_QUBITS,MAX_QUBITS+1,STEP):
#     times[str(i)]=np.median(times[str(i)])
#     memory[str(i)]=np.median(memory[str(i)])
# save_data_to_file({"simulator":"qiskit_aer_mps","state":"GHZ","system":SYSTEM,"min_qubits":MIN_QUBITS,"max_qubits":MAX_QUBITS,"step_qubits":STEP,"iter":ITER,"times":times,"memory":memory},f"qiskit/ghz_{SYSTEM}_({MIN_QUBITS}_{MAX_QUBITS}_{STEP})_{ITER}.json")
