from typing import List, Union
import string

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

INDICES = string.ascii_lowercase

def contract_tensors(A, B):
    ...

def outer_prod(v, u):
    ...

def init_state(n: int = 1, init_arr: List[float] = None)->np.array:
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
    
    state_list = [zero_qbit]*n

    if init_arr is not None:
        state_list = [np.array([1-x, x]) for x in init_arr]

    state = state_list[0]

    for qbit in state_list[1:]:
        state = np.tensordot(state, qbit, axes=-1)

    return state

#GATES


def CNOT()->np.array:
    C = np.array([[0, 0], [0, 1]])
    I = np.array([[1, 0], [0, 1]])
    SIG_X = np.array([[0, 1], [1, 0]])

    CNOT_1 = np.array([I-C, C])

    CNOT_2 = np.array([I, SIG_X])

    return np.einsum("ikj,lkm->ijlm", CNOT_1, CNOT_2)


def H()->np.array:
    return 1/np.sqrt(2)*np.array([[1,1],[1,-1]])

def X()->np.array:
    return np.array([[0, 1], [1, 0]])

def Y()->np.array:
    return np.array([[0, -1j], [1j, 0]])

def Z() -> np.array:
    return np.array([[1, 0], [0, -1]])


#APPLY GATES


def apply_one_qubit_gate(state:np.array,gate,qbit:int)->np.array:
    state=np.array(state) # perform deep copy
    # Number of dimensions of the state tensor
    num_dims = len(state.shape)-1
    if(qbit>num_dims):
        raise ValueError("qbit index bigger than dimension")
    
    qbit=num_dims-qbit

    #skip a and b chars
    input_subs = ''.join(chr(99 + i) for i in range(num_dims))

    # Replaces the k-th axis in input_subs with new index from gate_subs  
    input_subs = input_subs[:qbit] + 'a' + input_subs[qbit:]
    output_subs = input_subs[:qbit] + 'b' + input_subs[qbit+1:]

    # Construct the einsum string
    einsum_string = f'{input_subs},{"ab"}->{output_subs}'
    
    # Apply the gate using einsum
    return np.einsum(einsum_string, state, gate)






#PLOTS
def plot_nqbit_prob(MPS:np.array):
    n=len(MPS.shape)
    prob_distr = np.abs(MPS).__pow__(2)

    x_lbls = [''.join(state) for state in product('01', repeat=n)]
    probs = prob_distr.flatten()

    plt.bar(x_lbls, probs)
    plt.xlabel('Quantum States')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of nptRepr Quantum States')
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    state=init_state(2)
    print(state)
    plot_nqbit_prob(state)