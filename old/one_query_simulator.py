from utils import convert_to_einsum
from sql_commands import sql_einsum_query
from db_contraction import abstractDB,duckdb,sqlite3
from simulator import Simulator
import numpy as np
import utils as utils
from timeit import default_timer as timer

class OneQuerySimulator(Simulator):
    def __init__(self,num_qubits:int,gates:list,db:'abstractDB'=duckdb, use_timer:bool=True):
        self.use_timer=use_timer
        self.num_qubits=num_qubits
        self.gates=gates
        self.db_class=db
        self.state=None
        self.times=[]


        
    def simulate_one_query(self):
        db:abstractDB=self.db_class()
        einsum , indexes, prev=convert_to_einsum(self.num_qubits,self.gates)
        tensors=[np.array([1,0],dtype=np.complex128) for x in range(self.num_qubits)]
        tensors+=[getattr(utils,str(x[0]).upper())() for x in self.gates]
        tensorDict={}
        idx=[]
        for x,y in zip(tensors,indexes):
            tensorDict[y]=x
            idx.append(y)
        tic=timer()
        query=sql_einsum_query(einsum, idx, evidence=tensorDict, complex=True)
        res=db.contraction(query)
        toc=timer()
        if self.use_timer:
            self.times.append(toc-tic)

        self.state=self.convert_res_statevector(res,self.num_qubits)
        
        db.close()


    @staticmethod
    def convert_res_statevector(out:list,num_qubits:int):
        shape=[2]*num_qubits
        t=np.zeros(shape=shape,dtype=np.complex128)
        for x in out:
            t[x[:-2]]=x[-2]+x[-1]*1j
        return t.flatten()

    @staticmethod
    def run(num_qubits,gates,db:abstractDB):
        sim=OneQuerySimulator(num_qubits=num_qubits,gates=gates,db=db)
        sim.simulate_one_query()
        return sim
    
    def get_times(self):
        return np.array(self.times)

if __name__ == "__main__":
    q=OneQuerySimulator(3,[('h',1),('cnot',1,2)])
    q.simulate_one_query()
    print(q.times)