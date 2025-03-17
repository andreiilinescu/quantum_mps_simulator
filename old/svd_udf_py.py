import sqlite3
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

# Global reference to the connection (for demonstration)
GLOBAL_CONN = None
MAX_BOND=5
class SvdAggregator:
    def __init__(self):
        self.ti = []
        self.tj = []
        self.tk = []
        self.tl = []
        self.values = []


    def step(self, i, j,k,l, re,im, lf,rt):
        self.ti.append(i)
        self.tj.append(j)
        self.tk.append(k)
        self.tl.append(l)
        self.values.append(re+im*1j)
        self.l=lf
        self.r=rt

       

    def finalize(self):
        if not self.values:
            return None
        l=self.l
        r=self.r

        n=len(self.values)
        rows=np.zeros(n)
        cols=np.zeros(n)
        for i in range(len(self.values)):
            rows[i]=2*self.ti[i]+self.tj[i]
            cols[i]=r*self.tk[i]+self.tl[i]
        # Construct the sparse matrix
        try:
            M = coo_matrix(
                (self.values, (rows, cols)),
                shape=(2*l, 2*r)
            )
        except  Exception as e:
            print(e)

        U, S, Vh = np.linalg.svd(M.toarray(), full_matrices=False)

        if len(S)>MAX_BOND:
            U = U[:, :MAX_BOND]
            S = S[:MAX_BOND]
            Vh = Vh[:MAX_BOND, :]
        S=np.diag(S)


        U=U.reshape(l,2,int(U.size//l//2))

        Vh = S @ Vh
        Vh=Vh.reshape(int(Vh.size//r//2),2,r)
        # Create TEMP tables for U and V
        # Drop them first if they exist (just in case)
        # try:
        #     GLOBAL_CONN.execute("DROP TABLE IF EXISTS U")
        #     GLOBAL_CONN.execute("DROP TABLE IF EXISTS V")
        # except Exception as e: 
        #     print(e)
        # GLOBAL_CONN.execute("CREATE TEMP TABLE U (l INTEGER, ph INTEGER, r INTEGER, value REAL)")
        # GLOBAL_CONN.execute("CREATE TEMP TABLE V (l INTEGER, ph INTEGER, r INTEGER, value REAL)")

        # # Insert U
        # for idx, v in np.ndenumerate(U):
        #         if v !=0.0:
        #             GLOBAL_CONN.execute("INSERT INTO U (l, ph, r,value) VALUES (?, ?, ?, ?)", (idx[0], idx[1], idx[2], v))

        # for idx, v in np.ndenumerate(Vh):
        #         if v !=0.0:
        #             GLOBAL_CONN.execute("INSERT INTO V (l, ph, r,value) VALUES (?, ?, ?, ?)", (idx[0], idx[1], idx[2], v))

        # Return S as a JSON array
        try:
            l=json.dumps({
            "U_re": U.real.tolist(),
            "U_im": U.imag.tolist(),
            "Vh_re": Vh.real.tolist(),
            "Vh_im": Vh.imag.tolist(),
            "sh":[(l,int(U.size//l//2)),(int(Vh.size//r//2),r)]
             })
        except Exception as e: 
            print(e)

        return l


if __name__ == "__main__":
    conn = sqlite3.connect(":memory:")
    GLOBAL_CONN = conn  # set the global connection

    # Example data
    conn.execute("CREATE TABLE tA (i INTEGER, j INTEGER, k INTEGER, l INTEGER, re REAL, im REAL)")
    conn.executemany("INSERT INTO tA (i, j, k, l, re, im) VALUES (?, ?, ?, ?, ? , ?)",
                     [(0,0,0,0,0.72424289,0.0),
                      (0,1,1,0,0.72424289,0.0)])
    conn.commit()

    # Register the aggregate function
    conn.create_aggregate("svd_agg", 6, SvdAggregator)

    # Use it in a query
    cursor = conn.execute("SELECT svd_agg(i, j, k, l, re,im) FROM tA")
    result_json = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT svd_agg(i, j, k, l, re,im) FROM tA")
    result_json = cursor.fetchone()[0]
    conn.commit()
    print("Singular values (S):", result_json)
    # The U and V matrices are now in TEMP tables U and V
    # Check them:
    # print("U table:")
    # for row in conn.execute("SELECT * FROM U"):
    #     print(row)

    # print("V table:")
    # for row in conn.execute("SELECT * FROM V"):
    #     print(row)


    x=np.array([[0.72424289,0],[0,0.72424289]])
 
    u,s,v=np.linalg.svd(x)
    print(s)
    print(u)
    print(v)