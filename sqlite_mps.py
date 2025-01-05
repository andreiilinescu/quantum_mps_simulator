import sqlite3
import utils
import numpy as np
import json
from svd_udf import SvdAggregator
class SQL_MPS:
      def __init__(self,qbits:int=2,gates:set|None=None):
            self.num_qbits=qbits
            self.conn = sqlite3.connect(":memory:")
            self.conn.create_aggregate("svd_agg", 8, SvdAggregator)
            self.conn.execute("CREATE TABLE tTemp (i INTEGER, j INTEGER, k INTEGER, l INTEGER, re REAL, im REAL)")
            self.conn.execute("CREATE TABLE tShape (qbit INTEGER, left INTEGER, right INTEGER)")
            self.conn.execute("CREATE TABLE tOut (i INTEGER, j INTEGER, k INTEGER, re REAL, im REAL)")
            if gates is None:
                  gates={'H','X','Y','CNOT'}
            self.gates=gates
            self.initialize_db()
            self.init_gates()

      def initialize_db(self):   
            for i in range(self.num_qbits):
                  self.conn.execute(f"CREATE TABLE t{i} (i INTEGER, j INTEGER, k INTEGER, re REAL, im REAL)")
                  self.conn.execute(f"INSERT INTO t{i} VALUES (0,0,0,1,0)")
                  self.conn.execute(f"INSERT INTO tShape VALUES ({i},1,1)")
      
      def init_gates(self):
            for x in self.gates:
                  gate:np.ndarray=getattr(utils,x)()
                  if len(gate.shape)==2:
                        self.conn.execute(f"CREATE TABLE t{x} (i INTEGER, j INTEGER, re REAL, im REAL)")
                        gate:np.ndarray=getattr(utils,x)()
                        for idx,z in np.ndenumerate(gate):
                              self.conn.execute(f"INSERT INTO t{x} (i, j, re,im) VALUES ({idx[0]},{idx[1]} , {z.real}, {z.imag})")
                  elif len(gate.shape)==4:
                        self.conn.execute(f"CREATE TABLE t{x} (i INTEGER, j INTEGER, k INTEGER, l INTEGER, re REAL, im REAL)")
                        gate:np.ndarray=getattr(utils,x)()
                        for idx,z in np.ndenumerate(gate):
                              self.conn.execute(f"INSERT INTO t{x} (i, j,k,l, re,im) VALUES ({idx[0]},{idx[1]},{idx[2]},{idx[3]} , {z.real}, {z.imag})")
      
      def apply_one_qbit_gate(self,qbit:int,gate:str):
            res=self.conn.execute(f"""SELECT qb.i as i, gate.j as j, qb.k as k, SUM(gate.re * qb.re - gate.im * qb.im) AS re, SUM(gate.re * qb.im + gate.im * qb.re) AS im 
                                 FROM t{qbit} as qb JOIN  t{gate} gate ON qb.j= gate.i GROUP BY qb.i,gate.j,qb.k ORDER BY i,j,k""").fetchall()
            self.conn.execute(f"DELETE  FROM   t{qbit};")
            self.conn.executemany(f"INSERT INTO t{qbit} (i,j,k,re,im) VALUES (?,?,?,?,?)",res)

      def apply_two_qbit_gate(self,first_qbit:int,gate:str):
            #contraction
           
            res=self.conn.execute(f"""
                                    WITH cont AS (SELECT A.i as i, A.j as j, B.j as k, B.k as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im 
                                          FROM t{first_qbit} as A JOIN  t{first_qbit+1} B ON A.k= B.i GROUP BY A.i,A.j,B.j,B.k ORDER BY i,j,k,l)
                                    INSERT  INTO tTemp (i,j,k,l,re,im) 
                                    SELECT A.i as i, B.k as j, B.l as k, A.l as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im
                                    FROM cont as A JOIN t{gate} as  B on A.j=B.i AND A.k=B.j GROUP BY A.i, B.k, B.l, A.l HAVING SUM(B.re * A.re - B.im * A.im)!=0 OR SUM(B.re * A.im + B.im * A.re)!=0  ORDER BY i,j,k,l  
                                          """).fetchall()
            #clear tables
            self.conn.execute(f"DELETE  FROM   t{first_qbit};")
            self.conn.execute(f"DELETE  FROM   t{first_qbit+1};")
            l=self.conn.execute(f"SELECT left from tShape WHERE qbit={first_qbit}").fetchone()[0]
            r=self.conn.execute(f"SELECT right from tShape WHERE qbit={first_qbit+1}").fetchone()[0]
            #svd
            cursor = self.conn.execute(f"SELECT svd_agg(i, j, k, l, re, im, {l}, {r}) FROM tTemp")
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
                        self.conn.execute(f"INSERT INTO t{first_qbit+1} (i, j,k, re,im) VALUES ({idx[0]},{idx[1]},{idx[2]} , {v}, {v2})")

            sh=result_json["sh"]
            self.conn.execute(f"UPDATE tShape SET left={sh[0][0]}, right={sh[0][1]} WHERE qbit={first_qbit}")
            self.conn.execute(f"UPDATE tShape SET left={sh[1][0]}, right={sh[1][1]} WHERE qbit={first_qbit+1}")

            self.conn.execute(f"DELETE  FROM   tTemp;")
            self.conn.commit()
            #how to reshape matrix once this is done 2 by m where 2*m is the total dimension

      def check_db(self):
            for i in range(self.num_qbits):
                  res=self.conn.execute(f"SELECT * FROM t{i}").fetchall()
                  print(res)
            res=self.conn.execute(f"SELECT * FROM tTemp").fetchall()
            print(res)

      @staticmethod
      def run_circuit_json(path) -> 'SQL_MPS':
            file=open(path)
            data=json.load(file)
            num_qubits = data["number_of_qubits"]
            gates_data =data["gates"]
            gates =  {x['gate'] for x in gates_data}
            sim=SQL_MPS(num_qubits,gates)
            for x in gates_data:
                  # sim.check_db()
                  # print("\n")
                  if len(x['qubits'])==1:
                        sim.apply_one_qbit_gate(x['qubits'][0],x['gate'])
                  elif len(x['qubits'])==2:
                        sim.apply_two_qbit_gate(x['qubits'][0],x['gate'])
            return sim
      
      # def get_statevector(self):
      #       self.conn.execute("INSERT  INTO tOut (i,j,k,l,re,im) SELECT * FROM t{0}")
      #       for i in range(1,self.num_qbits):
      #             self.conn.execute(f"""WITH temp AS (SELECT A.i as i, A.j as j, B.j as k, B.k as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im 
      #                                     FROM tOut as A JOIN  t{i} B ON A.k= B.i GROUP BY A.i,A.j,B.j,B.k ORDER BY i,j,k,l)
      #                               INSERT  INTO tTemp (i,j,k,l,re,im)""")
           


t=SQL_MPS.run_circuit_json("./circuits/example.json")
# t.check_db()