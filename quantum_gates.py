import numpy as np

def create_two_qbit_control_gate(gate: np.array) -> np.ndarray:
    I=np.eye(4, dtype=np.complex128)
    I[2][2]=gate[0][0]
    I[2][3]=gate[0][1]
    I[3][2]=gate[1][0]
    I[3][3]=gate[1][1]
    return I.reshape(2,2,2,2)

# THREE QUBIT GATES

def TOFFOLI() -> np.ndarray:
    i=np.eye(8, dtype=np.complex128)
    i[6][6]=0
    i[6][7]=1
    i[7][7]=0
    i[7][6]=1
    # print(i)
    return i.reshape((2,2,2,2,2,2))


# TWO QUBIT GATES


def CH() -> np.ndarray:
    return create_two_qbit_control_gate(H())


def CNOT() -> np.ndarray:
    return create_two_qbit_control_gate(X())


def CX() -> np.ndarray:
    return create_two_qbit_control_gate(X())


def CY() -> np.ndarray:
    return create_two_qbit_control_gate(Y())


def CZ() -> np.ndarray:
    return create_two_qbit_control_gate(Z())


def CS() -> np.ndarray:
    return create_two_qbit_control_gate(S())


def CSDG() -> np.ndarray:
    return create_two_qbit_control_gate(SDG())


def CT() -> np.ndarray:
    return create_two_qbit_control_gate(T())


def CP(theta: float) -> np.ndarray:
    return create_two_qbit_control_gate(P(theta))

def CRY(theta: float) -> np.ndarray:
    return create_two_qbit_control_gate(RY(theta))

def CG(theta: float) -> np.ndarray:
    return create_two_qbit_control_gate(G(theta))


def SWAP() -> np.ndarray:
    swap=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    return swap.reshape((2,2,2,2))


# ONE QUBIT GATES


def H() -> np.ndarray:
    return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def X() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def Y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def Z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def S() -> np.ndarray:
    return np.array([[1, 0], [0, 1j]], dtype=np.complex128)


def SDG() -> np.ndarray:
    return np.array([[1, 0], [0, -1j]], dtype=np.complex128)


def T() -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def P(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)

def G(p: float) -> np.ndarray:
    return np.array([[complex(np.sqrt(p),0), complex(-np.sqrt(1-p),0)], [complex(np.sqrt(1-p),0), complex(np.sqrt(p),0)]])

def RY(angle) -> np.ndarray:
    return np.array([[np.cos(angle / 2.0), -np.sin(angle / 2.0)], [np.sin(angle / 2.0), np.cos(angle / 2.0)]])