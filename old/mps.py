import numpy as np
import utils as npt
import sql_commands as sqlc
from timeit import default_timer as timer
from db_contraction import duckDB
class MPS:
    def __init__(self, qbits: int,max_bond:int =5, use_timer: bool=True ,db_contraction=None):
        self.num_qubits = qbits
        self.max_bond=max_bond
        self.use_timer=use_timer 
        self.tensors=[np.array([1,0],dtype=np.complex128).reshape(1,2,1) for i in range(qbits)]
        self.times=[]

        if db_contraction:
            self._one_contraction=self._one_sql_contraction
            self._two_contraction=self._two_sql_contraction
            self.db_contraction=db_contraction
        else:
            self._one_contraction=self._one_numpy_contraction
            self._two_contraction=self._two_numpy_contraction



    def get_statevector(self):
        """Convert MPS back to full statevector (not efficient for large systems)"""
        tensor = self.tensors[0]
        for i in range(1, self.num_qubits):
            tensor = np.tensordot(tensor, self.tensors[i], axes=([-1], [0]))
        tensor = tensor.reshape(-1)
        return tensor
    
    def _one_numpy_contraction(self,qubit:int,gate:np.ndarray):
        self.tensors[qubit]=np.einsum("ijk,jl->ilk",self.tensors[qubit],gate)
    
    def _two_numpy_contraction(self, tensor:np.ndarray, gate:np.ndarray):
         return npt.apply_two_qubit_gate(tensor,gate,2,1)
    
    def _one_sql_contraction(self,qubit:int ,gate:np.ndarray):
        query=sqlc.sql_einsum_query("ijk,jl->ilk",['A','B'],{'A':self.tensors[qubit],'B':gate},complex=True)
        result=self.db_contraction(query)
        self.tensors[qubit]=np.zeros(self.tensors[qubit].shape,dtype=np.complex128)
        for x in result:
            self.tensors[qubit][x[0:-2]]=x[3]+x[4]*1j
    
    def _two_sql_contraction(self, tensor:np.ndarray, gate:np.ndarray):
        query=sqlc.sql_einsum_query("eabh,abcd->ecdh",['A','B'],{'A':tensor,'B':gate},complex=True)
        result=self.db_contraction(query)
        tensor=np.zeros(tensor.shape,dtype=np.complex128)
        for x in result:
            tensor[x[0:-2]]=x[-2]+x[-1]*1j
        return tensor
            
    def apply_one_qubit_gate(self, qubit:int, gate):
        """Apply one qubit gate"""
        if qubit>=len(self.tensors):
            raise ValueError("qubit number too large")
        #perform sql gate contraction
        tic=timer()
        self._one_contraction(qubit,gate)
        toc=timer()
        if self.use_timer:
            time=toc-tic
            self.times.append(time)
           

    def apply_two_qubit(self,qubit_one:int,qubit_two:int,gate):
        """Apply two qubit gate"""
        tic=timer()

        if qubit_one >= len(self.tensors) or qubit_two>=len(self.tensors):
            raise ValueError("qubit number too large")

        t1=self.tensors[qubit_one]
        t2=self.tensors[qubit_two]
        l=t1.shape[0]
        r=t2.shape[2]
        #contract tensors into one big tensor
        try:
            t=np.tensordot(t1, t2, axes=([-1], [0]))
        except:
            print(t1.shape)
            print(t2.shape)

        #perform gate contraction
        t=self._two_contraction(t,gate)
        #perform SVD
        t=t.reshape(l*2, r*2)
        U, S, Vh = np.linalg.svd(t, full_matrices=False)

        #Truncate SVD
        if len(S)>self.max_bond:
            U = U[:, :self.max_bond]
            S = S[:self.max_bond]
            Vh = Vh[:self.max_bond, :]
        S=np.diag(S)


        U=U.reshape(l,2,int(U.size//l//2))
       
        Vh = S @ Vh
        Vh=Vh.reshape(int(Vh.size//r//2),2,r)

        self.tensors[qubit_one]=U
        self.tensors[qubit_two]=Vh

        toc=timer()
        if self.use_timer:
            time=toc-tic
            self.times.append(time)

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

        # The final tensor should be Vh reshaped to (2, 2)
        final_tensor = Vh.reshape(2, 2, 1)
        mps_tensors.append(final_tensor)

        return mps_tensors
    
    def get_times(self) -> list:
        return self.times 


if __name__ == "__main__":
    m = MPS(3,max_bond=5)
    m.apply_one_qubit_gate(0,npt.H())
    m.apply_two_qubit(0,1,npt.CNOT())
    m.apply_two_qubit(1,2,npt.CNOT())

    print(m.get_statevector())
    
