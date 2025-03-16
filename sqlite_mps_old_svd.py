import sqlite3
import quantum_gates as quantum_gates
import numpy as np
import json
import tracemalloc
import pandas as pd
from timeit import default_timer as timer
from svd_udf_py import SvdAggregator
from plotting import plot_statevector
class SQLITE_MPS:
      def __init__(self,qbits:int,gates:dict):
            self.num_qbits=qbits
            self.times=[]
            self.conn = sqlite3.connect(":memory:")
            self.conn.create_aggregate("svd_agg", 8, SvdAggregator)
            gates["SWAP"]=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]).reshape((2,2,2,2))
            self.initialize_db()
            self.init_gates(gates)

      def initialize_db(self):   
            self.conn.execute("CREATE TABLE tShape (qbit INTEGER, left INTEGER, right INTEGER)")
            self.conn.execute("CREATE TABLE tOut (i INTEGER, j INTEGER, k INTEGER, re REAL, im REAL)")
            for i in range(self.num_qbits):
                  self.conn.execute(f"CREATE TABLE t{i} (i INTEGER, j INTEGER, k INTEGER, re REAL, im REAL)")
                  self.conn.execute(f"INSERT INTO t{i} VALUES (0,0,0,1,0)")
                  self.conn.execute(f"INSERT INTO tShape VALUES ({i},1,1)")
      
      def init_gates(self,gates):
            for name,gate in gates.items():
                  if len(gate.shape)==2: #create 1 qbit gate
                        self.conn.execute(f"CREATE TABLE t{name} (i INTEGER, j INTEGER, re REAL, im REAL)")
                        for idx,z in np.ndenumerate(gate):
                              if z!=0.0:
                                    self.conn.execute(f"INSERT INTO t{name} (i, j, re,im) VALUES ({idx[0]},{idx[1]} , {z.real}, {z.imag})")
                  elif len(gate.shape)==4: #create 2 qbit gate
                        self.conn.execute(f"CREATE TABLE t{name} (i INTEGER, j INTEGER, k INTEGER, l INTEGER, re REAL, im REAL)")
                        for idx,z in np.ndenumerate(gate):
                              if z!=0.0:
                                    self.conn.execute(f"INSERT INTO t{name} (i, j,k,l, re,im) VALUES ({idx[0]},{idx[1]},{idx[2]},{idx[3]} , {z.real}, {z.imag})")            
      
      def apply_one_qbit_gate(self,qbit:int,gate:str):
            res=self.conn.execute(f"""SELECT qb.i as i, gate.i as j, qb.k as k, SUM(gate.re * qb.re - gate.im * qb.im) AS re, SUM(gate.re * qb.im + gate.im * qb.re) AS im 
                                 FROM t{qbit} as qb JOIN  t{gate} gate ON qb.j= gate.j GROUP BY qb.i,gate.i,qb.k ORDER BY i,j,k""").fetchall()
            self.conn.execute(f"DELETE  FROM   t{qbit};")
            self.conn.executemany(f"INSERT INTO t{qbit} (i,j,k,re,im) VALUES (?,?,?,?,?)",res)

      def apply_two_qbit_gate(self,first_qbit:int,second_qubit:int,gate:str):
            if second_qubit-first_qbit==1:
                  self._two_qubit_contraction(first_qbit,first_qbit+1,gate)
            else:
                  path=[]
                  add=False
                  if(first_qbit>second_qubit): # inverse qbits if needed
                        first_qbit,second_qubit=second_qubit,first_qbit
                        add=True

                  for q in range(second_qubit, first_qbit+1, -1):
                        path.append((q - 1, q))

                  if add==True:
                        path.append((first_qbit,first_qbit+1))

                  for q1, q2 in path:
                        self._two_qubit_contraction(q1, q2, "SWAP")

                  self._two_qubit_contraction(first_qbit,first_qbit+1,gate)

                  for q1, q2 in reversed(path):
                        self._two_qubit_contraction(q1, q2, "SWAP")
      def _two_qubit_contraction(self,first_qbit:int,second_qubit:int,gate:str):
            cursor=self.conn.execute(f"""
                                    WITH cont AS (SELECT A.i as i, A.j as j, B.j as k, B.k as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im 
                                          FROM t{first_qbit} as A JOIN  t{second_qubit} B ON A.k= B.i GROUP BY A.i,A.j,B.j,B.k ),
                                     tTempq  AS (
                                    SELECT A.i as i, B.k as j, B.l as k, A.l as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im
                                    FROM cont as A JOIN t{gate} as  B on A.j=B.i AND A.k=B.j GROUP BY A.i, B.k, B.l, A.l HAVING SUM(B.re * A.re - B.im * A.im)!=0 OR SUM(B.re * A.im + B.im * A.re)!=0  
                                          )
                                    SELECT svd_agg(i,j,k,l,re,im, (SELECT "left"  FROM tShape WHERE qbit = {first_qbit}),(SELECT "right" FROM tShape WHERE qbit = {second_qubit}))
                                    FROM tTempq""")
            #clear tables
            self.conn.execute(f"DELETE  FROM   t{first_qbit};")
            self.conn.execute(f"DELETE  FROM   t{second_qubit};")
            result_json = json.loads(cursor.fetchone()[0])
            u_re=np.array(result_json["U_re"])
            u_im=np.array(result_json["U_im"])
            for idx, v in np.ndenumerate(u_re):
                  v2=u_im[idx]
                  if v !=0.0 or v2!=0.0:
                        self.conn.execute(f"INSERT INTO t{first_qbit} (i, j,k, re,im) VALUES ({idx[0]},{idx[1]},{idx[2]} , {v}, {v2})")
            v_re=np.array(result_json["Vh_re"])
            v_im=np.array(result_json["Vh_im"])
            for idx, v in np.ndenumerate(v_re):
                  v2=v_im[idx]
                  if v !=0.0 or v2!=0.0:
                        self.conn.execute(f"INSERT INTO t{second_qubit} (i, j,k, re,im) VALUES ({idx[0]},{idx[1]},{idx[2]} , {v}, {v2})")

            sh=result_json["sh"]
            self.conn.execute(f"UPDATE tShape SET left={sh[0][0]}, right={sh[0][1]} WHERE qbit={first_qbit}")
            self.conn.execute(f"UPDATE tShape SET left={sh[1][0]}, right={sh[1][1]} WHERE qbit={second_qubit}")


      def get_statevector_np(self):
            s=self.conn.execute("SELECT * FROM tShape").fetchall()
            tensors=[np.zeros((x[1],2,x[2]),dtype=np.complex128) for x in s]
            for i in range(self.num_qbits):
                  res=self.conn.execute(f"SELECT * FROM t{i}").fetchall()
                  for x in res:
                        tensors[i][x[0:3]]=x[3]+x[4]*1j

            tensor = tensors[0]
            for i in range(1, self.num_qbits):
                  tensor = np.tensordot(tensor, tensors[i], axes=([-1], [0]))
            tensor = tensor.reshape(-1)
            return tensor

      @staticmethod
      def run_circuit_json(data:dict) -> 'SQLITE_MPS':
            num_qubits = data["number_of_qubits"]
            gates_data =data["gates"]                  
            sim_gates= {}
            applied_gates=[]
            parametrized_gates={}
            for gate in gates_data:
                  name=gate['gate']
                  if "parameters" not in gate or len(gate["parameters"])==0:
                        sim_gates[name]=getattr(quantum_gates,name)()
                  else:
                        params=tuple(gate["parameters"]) 
                        parametrized_gates[name]=parametrized_gates.get(name,{})
                        if params not in parametrized_gates[name]:
                              parametrized_gates[name][params]=str(len(parametrized_gates[name]))
                        sim_gates[name+parametrized_gates[name][params]]=getattr(quantum_gates,name)(*params)
                        name+=parametrized_gates[name][params]
                  applied_gates.append((name,gate['qubits']))

            sim = SQLITE_MPS(num_qubits,sim_gates)

            for gate in applied_gates:
                  name, qbits=gate
                  tic=timer()
                  if len(qbits)==1:
                        sim.apply_one_qbit_gate(qbits[0],name)
                  elif len(qbits)==2:
                        sim.apply_two_qbit_gate(qbits[0],qbits[1],name)
                  toc=timer() 

                  sim.times.append(toc-tic)
            return sim


      

if __name__ == "__main__":
      file=open("./circuits/example.json")
      data=json.load(file)
      t=SQLITE_MPS.run_circuit_json(data)
      print(t.times)
      x=t.get_statevector_np()
      print(t.get_statevector_np())
      plot_statevector(x)
