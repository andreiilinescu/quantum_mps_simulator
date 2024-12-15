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
        self.rows = []
        self.cols = []
        self.values = []
        self.max_row = -1
        self.max_col = -1

    def step(self, row_idx, col_idx, val):
        self.rows.append(row_idx)
        self.cols.append(col_idx)
        self.values.append(val)
        if row_idx > self.max_row:
            self.max_row = row_idx
        if col_idx > self.max_col:
            self.max_col = col_idx

    def finalize(self):
        if not self.rows:
            return None

        # Construct the sparse matrix
        M = coo_matrix(
            (self.values, (self.rows, self.cols)),
            shape=(self.max_row+1, self.max_col+1)
        )

        # Choose a rank k for truncated SVD
        k = min(M.shape[0]-1, M.shape[1]-1, MAX_BOND)  # top 5 or less
        if k == 0:
            return None

        U, S, VT = svds(M, k=k)


        print(U)
        # Create TEMP tables for U and V
        # Drop them first if they exist (just in case)
        GLOBAL_CONN.execute("DROP TABLE IF EXISTS U")
        GLOBAL_CONN.execute("DROP TABLE IF EXISTS V")

        GLOBAL_CONN.execute("CREATE TEMP TABLE U (row_idx INTEGER, col_idx INTEGER, value REAL)")
        GLOBAL_CONN.execute("CREATE TEMP TABLE V (row_idx INTEGER, col_idx INTEGER, value REAL)")

        # Insert U
        rows_U, cols_U = U.shape
        for r in range(rows_U):
            for c in range(cols_U):
                val = float(U[r, c])
                if val !=0.0:
                    GLOBAL_CONN.execute("INSERT INTO U (row_idx, col_idx, value) VALUES (?, ?, ?)", (r, c, val))
        # Insert V (from VT)
        V = VT.T
        rows_V, cols_V = V.shape
        for r in range(rows_V):
            for c in range(cols_V):
                val = float(V[r, c])
                if val !=0.0:
                    GLOBAL_CONN.execute("INSERT INTO V (row_idx, col_idx, value) VALUES (?, ?, ?)", (r, c, val))

        # Return S as a JSON array
        return json.dumps({"S": S.tolist()})


if __name__ == "__main__":
    conn = sqlite3.connect(":memory:")
    GLOBAL_CONN = conn  # set the global connection

    # Example data
    conn.execute("CREATE TABLE tA (row INTEGER, col INTEGER, value REAL)")
    conn.executemany("INSERT INTO tA (row, col, value) VALUES (?, ?, ?)",
                     [(0,1,10.0),
                      (1,2,20.0),
                      (3,4,5.0)])
    conn.commit()

    # Register the aggregate function
    conn.create_aggregate("svd_agg", 3, SvdAggregator)

    # Use it in a query
    cursor = conn.execute("SELECT svd_agg(row, col, value) FROM tA")
    result_json = cursor.fetchone()[0]

    print("Singular values (S):", result_json)
    # The U and V matrices are now in TEMP tables U and V
    # Check them:
    print("U table:")
    for row in conn.execute("SELECT * FROM U"):
        print(row)

    print("V table:")
    for row in conn.execute("SELECT * FROM V"):
        print(row)
