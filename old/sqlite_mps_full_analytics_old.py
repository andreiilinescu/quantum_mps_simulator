import sqlite3
import quantum_gates as quantum_gates
import numpy as np
import json
import tracemalloc
import pandas as pd
from timeit import default_timer as timer
from svd_udf_py import SvdAggregator
from plotting import plot_statevector
# CPP=False
DEFAULT_GATES=["H","X","Y","Z","S","SDG","T","CH","CNOT","CX","CY","CZ","CS","CSDG","CT"]
class SQLITE_MPS_Analytics:
      def __init__(self,qbits:int,gates:dict):
            self.num_qbits=qbits
            self.analytics=[]
            self.conn = sqlite3.connect(":memory:")
            # if CPP:
            #       self.conn.enable_load_extension(True)
            #       self.conn.load_extension("./svd_udf.so")
            # else:
            self.conn.create_aggregate("svd_agg", 8, SvdAggregator)

            
            self.gates=gates
            self.gates["SWAP"]=None
            self.initialize_db()
            self.init_gates()

      def init_default_gates(self):
            for gate_name in DEFAULT_GATES:
                  gate:np.ndarray=getattr(quantum_gates,gate_name)()
                  self._setup_gate(gate_name,gate)

      def initialize_db(self):   
            self.conn.execute("CREATE TABLE tTemp (i INTEGER, j INTEGER, k INTEGER, l INTEGER, re REAL, im REAL)")
            self.conn.execute("CREATE TABLE tShape (qbit INTEGER, left INTEGER, right INTEGER)")
            self.conn.execute("CREATE TABLE tOut (i INTEGER, j INTEGER, k INTEGER, re REAL, im REAL)")
            for i in range(self.num_qbits):
                  self.conn.execute(f"CREATE TABLE t{i} (i INTEGER, j INTEGER, k INTEGER, re REAL, im REAL)")
                  self.conn.execute(f"INSERT INTO t{i} VALUES (0,0,0,1,0)")
                  self.conn.execute(f"INSERT INTO tShape VALUES ({i},1,1)")
      
      def init_gates(self):
            for name,params in self.gates.items():
                  try:
                        if params is None:
                              gate:np.ndarray=getattr(quantum_gates,name)()
                              self._setup_gate(name,gate)
                        else:
                             for val,suffix in params.items():
                                    gate:np.ndarray=getattr(quantum_gates,name)(*val) # pass multiple params if tuple
                                    self._setup_gate(name+str(suffix),gate)
                  except Exception as  e:
                              print(e)
                              print(f"!!----{name} gate not supported----!")
      
      def _setup_gate(self,x:str,gate:np.ndarray):
            if len(gate.shape)==2:
                  self.conn.execute(f"CREATE TABLE t{x} (i INTEGER, j INTEGER, re REAL, im REAL)")
                  for idx,z in np.ndenumerate(gate):
                        if z!=0.0:
                              self.conn.execute(f"INSERT INTO t{x} (i, j, re,im) VALUES ({idx[0]},{idx[1]} , {z.real}, {z.imag})")
            elif len(gate.shape)==4:
                  self.conn.execute(f"CREATE TABLE t{x} (i INTEGER, j INTEGER, k INTEGER, l INTEGER, re REAL, im REAL)")
                  for idx,z in np.ndenumerate(gate):
                        if z!=0.0:
                              self.conn.execute(f"INSERT INTO t{x} (i, j,k,l, re,im) VALUES ({idx[0]},{idx[1]},{idx[2]},{idx[3]} , {z.real}, {z.imag})")
      
      def apply_one_qbit_gate(self,qbit:int,gate:str):
            tic=timer()
            res=self.conn.execute(f"""SELECT qb.i as i, gate.i as j, qb.k as k, SUM(gate.re * qb.re - gate.im * qb.im) AS re, SUM(gate.re * qb.im + gate.im * qb.re) AS im 
                                 FROM t{qbit} as qb JOIN  t{gate} gate ON qb.j= gate.j GROUP BY qb.i,gate.i,qb.k ORDER BY i,j,k""").fetchall()
            toc=timer()
            self.analytics[-1]["pure_contraction"]=toc-tic
            tic=timer()
            self.conn.execute(f"DELETE  FROM   t{qbit};")
            self.conn.executemany(f"INSERT INTO t{qbit} (i,j,k,re,im) VALUES (?,?,?,?,?)",res)
            toc=timer()
            self.analytics[-1]["cleanup"]=toc-tic

      def apply_two_qbit_gate(self,first_qbit:int,second_qubit:int,gate:str):
            if second_qubit-first_qbit==1:
                  self._two_qubit_contraction(first_qbit,first_qbit+1,gate)
            else:
                  tic=timer()
                  path=[]
                  if(first_qbit>second_qubit):
                        first_qbit,second_qubit=second_qubit,first_qbit
                        path.append((first_qbit,first_qbit+1))

                  for q in range(second_qubit, first_qbit+1, -1):
                        path.append((q - 1, q))
                  #contraction
                  for q1, q2 in path:
                        self._two_qubit_contraction(q1, q2, "SWAP")
                  toc=timer()
                  self.analytics[-1]["swap_prep"]=toc-tic
                  self._two_qubit_contraction(first_qbit,first_qbit+1,gate)
                  tic=timer()
                  for q1, q2 in reversed(path):
                        self._two_qubit_contraction(q1, q2, "SWAP")
                  toc=timer()
                  self.analytics[-1]["swap_prep"]+=toc-tic
      def _two_qubit_contraction(self,first_qbit:int,second_qubit:int,gate:str):
            tic=timer()
            self.conn.execute(f"""
                                    WITH cont AS (SELECT A.i as i, A.j as j, B.j as k, B.k as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im 
                                          FROM t{first_qbit} as A JOIN  t{second_qubit} B ON A.k= B.i GROUP BY A.i,A.j,B.j,B.k ORDER BY i,j,k,l)
                                    INSERT  INTO tTemp (i,j,k,l,re,im) 
                                    SELECT A.i as i, B.k as j, B.l as k, A.l as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im
                                    FROM cont as A JOIN t{gate} as  B on A.j=B.i AND A.k=B.j GROUP BY A.i, B.k, B.l, A.l HAVING SUM(B.re * A.re - B.im * A.im)!=0 OR SUM(B.re * A.im + B.im * A.re)!=0  ORDER BY i,j,k,l  
                                          """)
            #clear tables
            toc=timer()
            self.analytics[-1]["pure_contraction"]=toc-tic

            tic=timer()
            cursor = self.conn.execute(f"""
                  SELECT svd_agg(i,j,k,l,re,im, (SELECT "left"  FROM tShape WHERE qbit = {first_qbit}),(SELECT "right" FROM tShape WHERE qbit = {second_qubit}))
                  FROM tTemp
                  """)
            toc=timer()
            self.analytics[-1]["svd"]=toc-tic

            tic=timer()
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

            self.conn.execute(f"DELETE  FROM   tTemp;")
            toc=timer()
            self.analytics[-1]["cleanup"]=toc-tic

      def check_db(self):
            for i in range(self.num_qbits):
                  res=self.conn.execute(f"SELECT * FROM t{i}").fetchall()
                  print(res)
            res=self.conn.execute(f"SELECT * FROM tTemp").fetchall()
            print(res)

      def save_state_tables(self):
            out=[]
            for i in range(self.num_qbits):
                  df = pd.read_sql_query(f"SELECT * FROM t{i};", self.conn)
                  print(f"t{i}")
                  out.append(df)
            return out
      
      def cleanup(self):
            for i in range(self.num_qbits):
                  self.conn.execute(f"DROP TABLE [IF EXISTS] t{i};")
            self.conn.execute("DROP TABLE [IF EXISTS] tShape;")
            self.conn.execute("DROP TABLE [IF EXISTS] tOut;")
            self.conn.execute("DROP TABLE [IF EXISTS] tTemp;")

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
      def run_circuit_json(data) -> 'SQLITE_MPS':
            num_qubits = data["number_of_qubits"]
            gates_data =data["gates"]                  
            gates= {}
            #setup gate dicts for parametrized gates
            for x in gates_data:
                  if "parameters" not in x or len(x["parameters"])==0:
                        gates[x['gate']]=None
                  else:
                        params=tuple(x["parameters"])
                        tmp=gates.get(x['gate'],{})
                        tmp[params]=len(tmp.keys())
                        gates[x['gate']]=tmp
            sim=SQLITE_MPS_Analytics(num_qubits,gates)
            #apply gates
            for x in gates_data:
                  gate_name=x['gate']
                  if "parameters" in x and len(x["parameters"])!=0:
                        val=tuple(x['parameters'])
                        gate_name+=str(gates[x['gate']][val])

                  sim.analytics.append({"total_time":None,"total_mem":None,**x})
                  tic=timer()

                  if len(x['qubits'])==1:
                        sim.apply_one_qbit_gate(x['qubits'][0],gate_name)
                  elif len(x['qubits'])==2:
                        sim.apply_two_qbit_gate(x['qubits'][0],x['qubits'][1],gate_name)

                  toc=timer() #timer


                  sim.analytics[-1]["total_time"]=toc-tic
                  sim.analytics[-1]["total_mem"]=mem
            return sim
      

if __name__ == "__main__":
      file=open("./circuits/example.json")
      data=json.load(file)
      t=SQLITE_MPS_Analytics.run_circuit_json(data)
      print(t.analytics)
      x=t.get_statevector_np()
      print(t.get_statevector_np())
      plot_statevector(x)
