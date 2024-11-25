import duckdb as db
import numpy as np
from termcolor import colored
 
def create_tables():
    create_statement="""
    CREATE TABLE tX (i INTEGER, j INTEGER, val DOUBLE);
    CREATE TABLE tA (i INTEGER, j INTEGER, val DOUBLE);
    """# i from 0 to n-1, j from 0 to d-1 
    db.sql(create_statement)
def store_matrices(X):
     
    insert_statement="INSERT INTO tX VALUES "
    for idx, x in np.ndenumerate(X):
        if x !=0:
            insert_statement+=f"({idx[0]},{idx[1]},{x}),"
    insert_statement= insert_statement[:-1]+";"
    db.sql(insert_statement)
def compute_summation_matrices():
    #create tL and tQ temporary tables
    create_statement="CREATE TEMPORARY TABLE tL (j INTEGER, val DOUBLE);  CREATE TEMPORARY TABLE tQ (i INTEGER, j INTEGER, val DOUBLE);"
    dimensions="WITH Dimensions AS (SELECT MAX(j) + 1 AS D, MAX(i) + 1 AS N FROM tX)"
    l=f"""INSERT INTO tL (j, val) SELECT j,SUM(val) FROM tX GROUP BY j;"""
    q=f"""INSERT INTO tQ (i, j, val)
    SELECT
        tX1.j AS i,
        tX2.j AS j,
        SUM(tX1.val * tX2.val) AS val
    FROM
        tX tX1
    JOIN
        tX AS tX2
        ON tX1.i = tX2.i
    GROUP BY
        tX1.j, tX2.j;
    """
    summation_query=create_statement+dimensions+l+q

    
    db.sql(summation_query)
    
def calculate_correlation_matrix():
    correlation_query=f"""
    WITH Dimensions AS (
        SELECT MAX(j) + 1 AS D, MAX(i) + 1 AS N
        FROM tX
    ),
    NormalizationFactors AS (
        SELECT
            tL1.j AS a,
            tL2.j AS b,
            SQRT(d.N * tQ1.val - tL1.val * tL1.val) AS denom_a,
            SQRT(d.N * tQ2.val - tL2.val * tL2.val) AS denom_b
        FROM
            tL tL1
        CROSS JOIN tL tL2
        LEFT JOIN tQ tQ1 ON tL1.j = tQ1.i AND tL1.j = tQ1.j -- Q_aa
        LEFT JOIN tQ tQ2 ON tL2.j = tQ2.i AND tL2.j = tQ2.j -- Q_bb
        CROSS JOIN Dimensions d
    ),
    Correlation AS (
        SELECT
            tQ.i AS a,
            tQ.j AS b,
            (d.N * tQ.val - tL1.val * tL2.val) /
            (NormalizationFactors.denom_a * NormalizationFactors.denom_b) AS val
        FROM
            tQ
        JOIN tL tL1 ON tQ.i = tL1.j
        JOIN tL tL2 ON tQ.j = tL2.j
        JOIN NormalizationFactors
        ON tQ.i = NormalizationFactors.a AND tQ.j = NormalizationFactors.b
        CROSS JOIN Dimensions d
        WHERE NormalizationFactors.denom_a > 0 AND NormalizationFactors.denom_b > 0 -- Avoid divide by zero
    )
    INSERT INTO tA (i,j,val)
    SELECT * FROM Correlation;
    DROP TABLE IF EXISTS tL;
    DROP TABLE IF EXISTS tQ;
    """
    db.sql(correlation_query)
def perform_householder():
    pass
def execute_qr():
    pass

def test_correlation_matrix_calc():
    create_tables()
    for x in range(100):
        A=np.random.rand(np.random.randint(2,5),np.random.randint(2,5))
        correct=np.corrcoef(np.transpose(A))

        store_matrices(A)
        compute_summation_matrices()
        calculate_correlation_matrix()

        out=np.reshape(np.array(db.sql("SELECT * FROM tA").fetchall())[:,2],correct.shape)
        
        db.sql("TRUNCATE TABLE tX; TRUNCATE TABLE tA;")
        if(np.allclose(out,correct) is False):
            print(colored("Correlation Calc Test Failed","red"))
            print(A)
            print(out)
            print(correct)

    print(colored("Correlation Calc Test Successful","green"))


if __name__ == "__main__":
    # A=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) 
    # print(A)
    # store_matrices(A)
    # compute_summation_matrices()
    # print(db.sql("SELECT * FROM tL").fetchall())
    # print(db.sql("SELECT * FROM tQ").fetchall())
    # calculate_correlation_matrix()
    # print(db.sql("SELECT * FROM tA").fetchall())
    # D,N,tmp=compute_summation_matrices()
    # tmp=tmp[0]
    # L=tmp[0:D]
    # Q=tmp[D:]
    # print(D)
    # print(N)
    # print(L)
    # print(Q)
    test_correlation_matrix_calc()