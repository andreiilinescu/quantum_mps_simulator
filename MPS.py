import numpy as np

class MPS:
    def __init__(self, qbits:int, initial_state:list | None = None):
        self.num_qubits=qbits
        if initial_state:
            self.tensors=self.statevector_to_mps(initial_state)
        else:
            s=[0]*(1<<qbits)
            s[0]=1
            self.tensors=self.statevector_to_mps(s)

    @staticmethod
    def statevector_to_mps(state_vector: list) :
        mps_tensors = []

        state_vector=np.array(state_vector)
        num_qubits=int(np.log2(state_vector.size))
        dim=int(state_vector.size//2)
        
        # Reshape state vector for the first SVD (split first qubit from the rest)
        state_matrix = state_vector.reshape(2, dim)
        print(state_matrix)
        for i in range(num_qubits - 1):
            # Perform SVD
            U, S, Vh =np.linalg.svd(state_matrix, full_matrices=False)
            

            Lambda=np.sqrt(np.diag(S))
            
            # Store the U matrix (as the MPS tensor for the current qubit)
            mps_tensors.append(U)

            # Store the singular values (transform vector to diag matrix)
            Vh=Lambda@Vh

            # Reshape Vh for the next iteration
            dim //= 2
            if i<num_qubits-2:
                state_matrix = Vh.reshape(2, 2, dim)

        # The final tensor should be Vh reshaped to (2, 2)
        final_tensor = Vh.reshape(2,2)
        mps_tensors.append(final_tensor)

        return mps_tensors
    
    def get_statevector(self):
        # Convert MPS back to full statevector (not efficient for large systems)
        tensor = self.tensors[0]
        for i in range(1, self.num_qubits):
            tensor = np.tensordot(tensor, self.tensors[i], axes=([-1], [0]))
        tensor = tensor.reshape(-1)
        return tensor
    
if __name__ == "__main__":
    m=MPS(3)
    print(m.tensors)
    print(m.get_statevector())