import numpy as np
import nptRepr as npt

class MPS:
    def __init__(self, qbits: int,max_bond:int =5 , initial_state: list | None = None):
        self.num_qubits = qbits
        self.max_bond=max_bond
        if initial_state:
            self.tensors = self.statevector_to_mps(initial_state)
        else:
            self.tensors=[np.array([1,0],dtype=np.complex128).reshape(1,2,1) for i in range(qbits)]


    @staticmethod
    def statevector_to_mps(state_vector: list):
        mps_tensors = []

        state_vector = np.array(state_vector)
        num_qubits = int(np.log2(state_vector.size))
        dim = int(state_vector.size // 2)

        # Reshape state vector for the first SVD (split first qubit from the rest)
        state_matrix = state_vector.reshape(2, dim)
        for i in range(num_qubits - 1):
            # Perform SVD
            U, S, Vh = np.linalg.svd(state_matrix, full_matrices=False)

            Lambda = np.sqrt(np.diag(S))

            # Store the U matrix (as the MPS tensor for the current qubit)
            if i != 0:
                U = np.stack([U, U], axis=2)
            else:
                U=U.reshape(1,2,2)
            U = U @ Lambda
            mps_tensors.append(U)

            # Store the singular values (transform vector to diag matrix)
            Vh = Lambda @ Vh

            # Reshape Vh for the next iteration
            dim = dim // 2
            if i < num_qubits - 2:
                # remove decomposed qubit (reduce dimension)
                state_matrix = Vh.reshape(2, dim, 2).sum(axis=2)
                print(state_matrix)

        # The final tensor should be Vh reshaped to (2, 2)
        final_tensor = Vh.reshape(2, 2, 1)
        mps_tensors.append(final_tensor)

        return mps_tensors

    def get_statevector(self):
        # Convert MPS back to full statevector (not efficient for large systems)
        tensor = self.tensors[0]
        for i in range(1, self.num_qubits):
            tensor = np.tensordot(tensor, self.tensors[i], axes=([-1], [0]))
        tensor = tensor.reshape(-1)
        return tensor

    def apply_one_qubit_gate(self, qubit:int, gate):
        if qubit>=len(self.tensors):
            raise ValueError("qubit number too large")
        self.tensors[qubit]=np.einsum("ijk,jl->ilk",self.tensors[qubit],gate)

    def apply_two_qubit(self,qubit_one:int,qubit_two:int,gate):
        if qubit_one >= len(self.tensors) or qubit_two>=len(self.tensors):
            raise ValueError("qubit number too large")
        
        t1=self.tensors[qubit_one]
        t2=self.tensors[qubit_two]
        l=t1.shape[0]
        r=t2.shape[2]

        #contract tensors into one big tensor
        t=np.tensordot(t1, t2, axes=([-1], [0]))
        t=npt.apply_two_qubit_gate(t,gate,1,2)


        #perform SVD
        t=t.reshape(l*2, r*2)
        
        U, S, Vh = np.linalg.svd(t, full_matrices=False)
        print(S)

        if len(S)>self.max_bond:
            k=len(S)-self.max_bond-1
            idx=np.argpartition(S,k)[:k+1]
            print(idx)
            S=S[~idx]
            print(S)
            U = U[:,~idx]
            Vh = Vh[~idx,:]

        S=np.diag(S)


        U=U.reshape(l,2,int(U.size//l//2))
        
        Vh = S @ Vh
        Vh=Vh.reshape(int(Vh.size//r//2),2,r)

        self.tensors[qubit_one]=U
        self.tensors[qubit_two]=Vh

        

if __name__ == "__main__":
    m = MPS(2,max_bond=3)
    m.apply_one_qubit_gate(1,npt.H())
    m.apply_two_qubit(0,1,npt.CNOT())
    # m.apply_two_qubit(0,1,npt.CNOT())
    print("\n")
    for x in m.tensors:
        print(x.shape)
    print(m.get_statevector())
    
