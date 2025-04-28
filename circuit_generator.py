import numpy as np

def generate_ghz_circuit(num_qubits:int) -> dict:
    data={ "number_of_qubits":num_qubits,"gates":[]}
    data["gates"].append({ "qubits": [0],"gate": "H"})
    for i in range(num_qubits-1):
        data['gates'].append({ "qubits": [i, i+1],"gate": "CNOT"})
    return data

def generate_qft_circuit(num_qubits:int) -> dict:
    data={ "number_of_qubits":num_qubits,"gates":[]}
    for i in range(num_qubits):
        data["gates"].append({ "qubits": [i],"gate": "H"})
        for j in range(i + 1, num_qubits):
            data['gates'].append({ "qubits": [i, j],"gate": "CP","parameters":[np.pi / (2 ** (j - i))]})
    return data