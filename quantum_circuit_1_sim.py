import numpy as np
import matplotlib.pyplot as plt
# Define the quantum gates

# Hadamard gate for 1 qubit
H = (1 / np.sqrt(2)) * np.array([[1, 1], 
                                 [1, -1]])
# CNOT gate for 2 qubits (control on qubit 0, target on qubit 1)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

# Initial state: |00> (tensor product of two qubits)
q0=np.array([1,0])
initial_state = np.tensordot(q0,q0,axes=0)

# Apply Hadamard to the first qubit
state_after_H = np.tensordot(H,initial_state,axes=1)
print(state_after_H)
# Apply the CNOT gate
# print(final_state)
final_state=state_after_H.flatten()
final_state=np.tensordot(CNOT,final_state,axes=1)
final_state=final_state.reshape((2,2))
# final_state=final_state[0:2,0:2]
# Print the resulting state vector
print("Final state vector after applying the Hadamard and CNOT gates:")
print(final_state)

probs=np.square(final_state)


plt.bar(["|00>","|01>","|10>","|11>"],probs.flatten())

plt.show()