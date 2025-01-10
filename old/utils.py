from typing import List, Union
import string

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

INDICES = ""
valid_characters = [(65, 65+26), (97, 97+26), (192, 214), (216, 246), (248, 328), (330, 383), (477, 687), (913, 974), (1024, 1119)]
for tup in valid_characters:    
    for i in range(tup[0], tup[1]):        
        INDICES += chr(i)


def init_state(n: int = 1, init_arr: List[float] = None) -> np.array:
    """
    creates a tensor for the initial state of a given number of qubits

    input:
        n: number of qubits in the quantum circuit
        init_arr: list of float values for each qubit in the circuit
    returns:
        state: numpy tensor of the initial state as the outer product of every qubit state
    """
    # Start with a simple 2 x 2^(n-1) matrix that represents the initial state
    zero_qbit = np.array([1, 0])

    state_list = [zero_qbit] * n

    if init_arr is not None:
        state_list = [np.array([1 - x, x]) for x in init_arr]

    state = state_list[0]

    for qbit in state_list[1:]:
        state = np.tensordot(state, qbit, axes=0)

    return state


# GATES


def create_two_qbit_control_gate(gate: np.array) -> np.array:
    I=np.eye(4, dtype=np.complex128)
    I[2][2]=gate[0][0]
    I[2][3]=gate[0][1]
    I[3][2]=gate[1][0]
    I[3][3]=gate[1][1]
    return I.reshape(2,2,2,2)

# THREE QUBIT GATES

def TOFFOLI() -> np.array:
    i=np.eye(8, dtype=np.complex128)
    i[6][6]=0
    i[6][7]=1
    i[7][7]=0
    i[7][6]=1
    # print(i)
    return i.reshape((2,2,2,2,2,2))


# TWO QUBIT GATES


def CH() -> np.array:
    return create_two_qbit_control_gate(H())


def CNOT() -> np.array:
    return create_two_qbit_control_gate(X())


def CX() -> np.array:
    return create_two_qbit_control_gate(X())


def CY() -> np.array:
    return create_two_qbit_control_gate(Y())


def CZ() -> np.array:
    return create_two_qbit_control_gate(Z())


def CS() -> np.array:
    return create_two_qbit_control_gate(S())


def CSDG() -> np.array:
    return create_two_qbit_control_gate(SDG())


def CT() -> np.array:
    return create_two_qbit_control_gate(T())


def CP(theta: float) -> np.array:
    return create_two_qbit_control_gate(P(theta))


def SWAP() -> np.array:
    swap=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    return swap.reshape((2,2,2,2))


# ONE QUBIT GATES


def H() -> np.array:
    return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def X() -> np.array:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def Y() -> np.array:
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def Z() -> np.array:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def S() -> np.array:
    return np.array([[1, 0], [0, 1j]], dtype=np.complex128)


def SDG() -> np.array:
    return np.array([[1, 0], [0, -1j]], dtype=np.complex128)


def T() -> np.array:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def P(theta: float) -> np.array:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)


# APPLY GATES


def apply_one_qubit_gate(state: np.array, gate: np.array, qbit: int) -> np.array:
    state = np.array(state)  # perform deep copy
    # Number of dimensions of the state tensor
    num_dims = len(state.shape)
    if qbit > num_dims:
        raise ValueError("qbit index bigger than dimension")
    if gate.shape != (2, 2):
        raise ValueError("wrong gate shape")

    axis = num_dims - qbit - 1  # convert to qiskit qbit numbering

    # skip a and b chars
    indices = [chr(99 + i) for i in range(num_dims)]
    input_subs = "".join(indices)
    output_subs = "".join(indices)

    # Replaces the k-th axis in input_subs with new index from gate_subs
    input_subs_list = list(input_subs)
    output_subs_list = list(output_subs)
    input_subs_list[axis] = "a"
    output_subs_list[axis] = "b"
    input_subs = "".join(input_subs_list)
    output_subs = "".join(output_subs_list)

    # Construct the einsum string
    einsum_string = f'{input_subs},{"ab"}->{output_subs}'
    # print(einsum_string)

    # Apply the gate using einsum
    return np.einsum(einsum_string, state, gate)


def apply_two_qubit_gate(
    state: np.array, gate: np.array, qbit_one: int, qbit_two: int, mps=False
) -> np.array:
    num_dims = len(state.shape)
    if qbit_one > num_dims or qbit_two > num_dims:
        raise ValueError("qbit index bigger than dimension")
    if gate.shape != (2, 2, 2, 2):
        raise ValueError("wrong gate shape")
    axis1 = num_dims - qbit_one - 1
    axis2 = num_dims - qbit_two - 1
    if mps==True:
        axis1=qbit_one
        axis2=qbit_two
    # Build indices for einsum
    indices = [chr(101 + i) for i in range(num_dims)]
    input_subs = "".join(indices)
    output_subs = "".join(indices)

    # Replace indices at axis1 and axis2
    input_subs_list = list(input_subs)
    output_subs_list = list(output_subs)

    input_subs_list[axis1] = "a"
    input_subs_list[axis2] = "b"
    output_subs_list[axis1] = "c"
    output_subs_list[axis2] = "d"

    input_subs = "".join(input_subs_list)
    output_subs = "".join(output_subs_list)

    einsum_str = f"{input_subs},abcd->{output_subs}"
    # Apply the gate using einsum
    # print(einsum_str)
    return np.einsum(einsum_str, state, gate)


def apply_three_qubit_gate(
    state: np.array, gate: np.array, qbit_one: int, qbit_two: int, qbit_three:int
) -> np.array:
    num_dims = len(state.shape)
    if qbit_one > num_dims or qbit_two > num_dims or qbit_three>num_dims:
        raise ValueError("qbit index bigger than dimension")
    if gate.shape != (2, 2, 2, 2, 2, 2):
        raise ValueError("wrong gate shape")
    axis1 = num_dims - qbit_one - 1
    axis2 = num_dims - qbit_two - 1
    axis3 = num_dims - qbit_three - 1
    # Build indices for einsum
    indices = [chr(103 + i) for i in range(num_dims)]
    input_subs = "".join(indices)
    output_subs = "".join(indices)

    # Replace indices at axis1 and axis2
    input_subs_list = list(input_subs)
    output_subs_list = list(output_subs)

    input_subs_list[axis1] = "a"
    input_subs_list[axis2] = "b"
    input_subs_list[axis3] = "c"
    output_subs_list[axis1] = "d"
    output_subs_list[axis2] = "e"
    output_subs_list[axis3] = "f"

    input_subs = "".join(input_subs_list)
    output_subs = "".join(output_subs_list)

    einsum_str = f"{input_subs},abcdef->{output_subs}"
    # Apply the gate using einsum
    return np.einsum(einsum_str, state, gate)

# PLOTS
def plot_nqbit_prob_old(MPS: np.array):
    n = len(MPS.shape)
    prob_distr = np.abs(MPS).__pow__(2)

    x_lbls = ["".join(state) for state in product("01", repeat=n)]
    probs = prob_distr.flatten()

    plt.bar(x_lbls, probs)
    plt.xlabel("Quantum States")
    plt.ylabel("Probability")
    plt.title("Probability Distribution of nptRepr Quantum States")
    plt.ylim(0, 1)
    plt.show()

def convert_to_einsum(num_qubits,gates):        
    einsum_notation = ""        
    current_out_index = [""]*num_qubits        
    index_sizes = {}        
    parameters = []

    # start by giving each qubit a single letter        
    for last_index in range(num_qubits):            
        einsum_notation += INDICES[last_index] + ","            
        current_out_index[last_index] = INDICES[last_index]            
        index_sizes[INDICES[last_index]] = 2            
        parameters.append("T0")        
    
    # now the gates are applied        
    for gate in gates:            
        last_index += 1            
        match len(gate):               
            case 3:                    
                einsum_notation += INDICES[last_index:last_index+2] + current_out_index[gate[1]] + current_out_index[gate[2]] + ","                    
                current_out_index[gate[1]] = INDICES[last_index]                    
                current_out_index[gate[2]] = INDICES[last_index+1]                    
                index_sizes[INDICES[last_index]] = 2                    
                last_index += 1                
            case 2:                    
                einsum_notation += INDICES[last_index] + current_out_index[gate[1]] + ","                    
                current_out_index[gate[1]] = INDICES[last_index]            
        parameters.append(gate[0])            
        index_sizes[INDICES[last_index]] = 2

    einsum_notation = einsum_notation[:-1]                
    
    last_index += 1        
    
    einsum_notation += "->" +  "".join(current_out_index)        
    return einsum_notation, index_sizes, parameters


if __name__ == "__main__":
    # print(TOFFOLI())
    # # state = init_state(3)
    # # print(state)
    # # state = apply_one_qubit_gate(state, H(), 0)
    # # state = apply_one_qubit_gate(state, H(), 2)
    # # state = apply_two_qubit_gate(state, CY(), 0, 1)
    # print(CP(np.pi / 2))

    # x=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,np.exp(np.pi/2)]])
    # print(x.reshape(2,2,2,2))
    print(INDICES)
