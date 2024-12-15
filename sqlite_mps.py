import sqlite3
import utils
import numpy as np
from scipy.sparse import coo_matrix

class SQL_MPS:
      def __init__(self,qbits:int=2):
            self.num_qbits=qbits
            self.conn = sqlite3.connect(":memory:")
            self.gates=['H','X','Y','CNOT']
            self.initialize_db()
            self.init_gates()

      def initialize_db(self):   
            for i in range(self.num_qbits):
                  self.conn.execute(f"CREATE TABLE t{i} (i INTEGER, j INTEGER, k INTEGER, re REAL, im REAL)")
                  self.conn.execute(f"INSERT INTO t{i} VALUES (0,0,0,1,0)")
      
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
            self.conn.execute("CREATE TEMPORARY TABLE tTemp (i INTEGER, j INTEGER, k INTEGER, l INTEGER, re REAL, im REAL)")
            res=self.conn.execute(f"""
                                    WITH cont AS (SELECT A.i as i, A.j as j, B.j as k, B.k as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im 
                                          FROM t{first_qbit} as A JOIN  t{first_qbit+1} B ON A.k= B.i GROUP BY A.i,A.j,B.j,B.k ORDER BY i,j,k,l)
                                    INSERT  INTO tTemp (i,j,k,l,re,im) 
                                    SELECT A.i as i, A.j as j, B.j as k, B.k as l, SUM(B.re * A.re - B.im * A.im) AS re, SUM(B.re * A.im + B.im * A.re) AS im
                                    FROM cont as A JOIN t{gate} B on A.j=B.i AND A.k=B.j GROUP BY A.i, A.j, B.j, B.k HAVING SUM(B.re * A.re - B.im * A.im)!=0 OR SUM(B.re * A.im + B.im * A.re)!=0  ORDER BY i,j,k,l  
                                          """).fetchall()
            
            print(res)

            #how to reshape matrix once this is done 2 by m where 2*m is the total dimension
      def check_db(self):
            for i in range(self.num_qbits):
                  res=self.conn.execute(f"SELECT * FROM t{i}").fetchall()
                  print(res)
            res=self.conn.execute(f"SELECT * FROM tTemp").fetchall()
            print(res)
            print()
           

s=SQL_MPS(2)
s.apply_one_qbit_gate(0,'H')
s.apply_two_qbit_gate(0,"CNOT")
s.check_db()
