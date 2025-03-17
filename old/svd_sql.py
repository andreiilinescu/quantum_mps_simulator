import duckdb as db
import numpy as np
from termcolor import colored
 
def create_tables():
    create_statement="""
    CREATE TABLE tX (i INTEGER, j INTEGER, val DOUBLE);
    CREATE TABLE tA (i INTEGER, j INTEGER, val DOUBLE);
    CREATE TABLE tU (i INTEGER, j INTEGER, val DOUBLE);
    CREATE TABLE tEye (i INTEGER, j INTEGER, val DOUBLE);
    """# i from 0 to n-1, j from 0 to d-1 

    db.sql(create_statement)
    insert_statement="INSERT INTO tEye VALUES "
    N=20
    for i in range (0,N):
        insert_statement+=f"({i},{i},1),"
    insert_statement= insert_statement[:-1]+";"
    db.sql(insert_statement)
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

def prep_householder():
    create_statement="""CREATE TEMPORARY TABLE tW (i INTEGER, val DOUBLE);  
    CREATE TEMPORARY TABLE tP (i INTEGER, j INTEGER, val DOUBLE); 
    CREATE TEMPORARY TABLE tH (i INTEGER, j INTEGER, val DOUBLE);
    CREATE TEMPORARY TABLE tT (i INTEGER, j INTEGER, val DOUBLE);
    CREATE TEMPORARY TABLE tAlphaR (alpha DOUBLE,r DOUBLE);
    """
    db.sql(create_statement)
    initialize_U="""
    WITH Dimensions AS (
            SELECT MAX(j) + 1 AS D
            FROM tX
    )
    INSERT INTO tU (i,j,val)
    SELECT tEye.i,tEye.j,tEye.val FROM tEye
    CROSS JOIN Dimensions d
    WHERE tEye.i<d.D;
    """
    initialize_H="""
    INSERT INTO tH (i,j,val)
    SELECT * FROM tA;
    """
    db.sql(initialize_U+initialize_H)

def get_d():
    return db.sql("SELECT MAX(j) + 1 FROM tX;").fetchall()[0][0]

def get_alpha_r(k):
    alpha_statement=f"""
        WITH SumOfSquares AS (
        SELECT
            SUM(POWER(val, 2)) AS sum_squares
        FROM
            tA
        WHERE
            i >= {k+1} -- row index >= k+1
            AND j = {k}  -- column index = k
        )
        SELECT
            -SIGN(v.val) * SQRT(s.sum_squares) AS alpha,
            SQRT(
                0.5 * POWER(-SIGN(v.val) * SQRT(s.sum_squares), 2) 
                - 0.5 * (-SIGN(v.val) * SQRT(s.sum_squares)) * v.val
            ) AS r
        FROM
            SumOfSquares s,
            tA v
        WHERE
            v.i = {k + 1} -- row index = k+1
            AND v.j = {k}; -- column index = k
    """
    return db.sql(alpha_statement).fetchall()[0]

def perform_householder():
    prep_householder()
    d=get_d()
    for k in range(0,d-2):
        db.sql("TRUNCATE TABLE tW;TRUNCATE TABLE tAlphaR;") #reset alpha and r
        alpha_statement=f"""
        WITH SumOfSquares AS (
            SELECT
                SUM(POWER(val, 2)) AS sum_squares
            FROM
                tH
            WHERE
                i >= {k + 1 }-- row index >= k+1
                AND j = {k}  -- column index = k
        ),
        AlphaAndR AS (
            SELECT
                -SIGN(v.val) * SQRT(s.sum_squares) AS alpha,
                SQRT(
                    0.5 * POWER(-SIGN(v.val) * SQRT(s.sum_squares), 2) 
                    - 0.5 * (-SIGN(v.val) * SQRT(s.sum_squares)) * v.val
                ) AS r
            FROM
                SumOfSquares s,
                tH v
            WHERE
                v.i = {k + 1} -- row index = k+1
                AND v.j = {k} -- column index = k
        )
        INSERT INTO tAlphaR (alpha, r)
        SELECT alpha, r
        FROM AlphaAndR;
        """
        alpha,r=db.sql(alpha_statement+"SELECT alpha, r From tAlphaR;").fetchall()[0] #recalculate alpha and r
        print(alpha)
        print(r)
        w_statement=f"""
        INSERT INTO tW (i, val)
        SELECT
            {k + 1} AS i, -- Row index is fixed as k+1
            (v.val - t.alpha) / (2 * t.r) AS w_value -- Compute w_(k+1)
        FROM
            tH v
            CROSS JOIN tAlphaR t
        WHERE
            v.i = {k + 1} -- Row index = k+1
            AND v.j = {k}; -- Column index = k;

        INSERT INTO tW (i, val)
        SELECT
            v.i AS i, -- Row index for w_j (j > k+1)
            v.val / (2 * t.r) AS w_value -- Compute w_j
        FROM
            tH v
            CROSS JOIN tAlphaR t
        WHERE
            v.i > {k + 1} -- Row index > k+1
            AND v.j = {k}; -- Column index = k;
        """
        db.sql(w_statement)
        print(db.sql("SELECT * FROM tW;"))
        p_statement=f"""
        INSERT INTO tP VALUES ({k},{k},1);
        INSERT INTO tP (i, j, val)
        SELECT
            wi.i AS i,
            wj.i AS j,
            CASE
                WHEN wi.i = wj.i THEN 1 - 2 * wi.val * wj.val
                ELSE -2 * wi.val * wj.val
            END AS value
        FROM
            tW wi, -- w_i values
            tW wj; -- w_j values;
        """
        db.sql(p_statement)
        v_full_multiply_statement = f"""
        INSERT INTO tT (i, j, val)
        SELECT
            pk.i AS i, -- Row index from P_k
            vk.j AS j, -- Column index from V_k
            SUM(pk.val * vk.val) AS val -- Matrix multiplication
        FROM
            tP pk -- P_k matrix
            JOIN tH vk ON pk.j = vk.i -- Matrix multiplication P_k * V_k
        GROUP BY
            pk.i, vk.j;

        TRUNCATE TABLE tH;

        -- Step 2: Compute (P_k * V_k) * P_k
        INSERT INTO tH (i, j, val)
        SELECT
            temp.i AS i, -- Row index from the intermediate result
            pk.j AS j, -- Column index from P_k
            SUM(temp.val * pk.val) AS val -- Matrix multiplication
        FROM
            tT temp -- Intermediate result (P_k * V_k)
            JOIN tP pk ON temp.j = pk.i -- Matrix multiplication (P_k * V_k) * P_k
        GROUP BY
            temp.i, pk.j;
        """
        db.sql(v_full_multiply_statement)
        u_update_statement = f"""
        TRUNCATE TABLE tT;

        -- Update U_k by multiplying U_{k-1} * P_k
        INSERT INTO tT (i, j, val)
        SELECT
            uk.i AS i, -- Row index from U_{k-1}
            pk.j AS j, -- Column index from P_k
            SUM(uk.val * pk.val) AS val -- Matrix multiplication
        FROM
            tU uk -- Previous U_{k-1}
            JOIN tP pk ON uk.j = pk.i -- Matrix multiplication U_{k-1} * P_k
        GROUP BY
            uk.i, pk.j;

        TRUNCATE TABLE tU;

        INSERT INTO tU (i,j,val)
        SELECT * FROM tT;
        """
        db.sql(u_update_statement)

    db.sql("TRUNCATE TABLE tA; INSERT INTO tA (i,j,val) SELECT * FROM tH;")
    db.sql("DROP TABLE IF EXISTS tW; DROP TABLE IF EXISTS tP; DROP TABLE IF EXISTS tH; DROP TABLE IF EXISTS tT;")
    
def execute_qr():
    db.sql("""CREATE TEMPORARY TABLE tT (i INTEGER,j INTEGER, val DOUBLE); CREATE TEMPORARY TABLE tQ (i INTEGER,j INTEGER, val DOUBLE); CREATE TEMPORARY TABLE tR (i INTEGER,j INTEGER, val DOUBLE);""")
    db.sql("INSERT INTO tT SELECT * FROM tA;")
    d=get_d()
    for i in range(d):
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

def test_alpha_calc():
    create_tables()
    A=np.random.rand(np.random.randint(2,5),np.random.randint(2,5))
    k=0
    correct=np.corrcoef(np.transpose(A))

    store_matrices(A)
    compute_summation_matrices()
    calculate_correlation_matrix()

    out=np.reshape(np.array(db.sql("SELECT * FROM tA").fetchall())[:,2],correct.shape)

    prep_householder()
    a,r=get_alpha_r(0)

    correct_a=-np.sign(correct[k+1][k])*np.sqrt(np.sum(correct[k+1:,k]**2))
    correct_r=np.sqrt(0.5*correct_a*correct_a-0.5*correct_a*correct[k+1][k])

    if(np.allclose(a,correct_a) is False or np.allclose(r,correct_r) is False):
            print(colored("Correlation Calc Test Failed","red"))
            print(A)
            print(out)
            print(correct)
            print((a,r))
            print((correct_a,correct_r))

    print(colored("Correlation Calc Test Successful","green"))

def calc_w(A,k):
    correct=np.corrcoef(np.transpose(A))
    correct_a=-np.sign(correct[k+1][k])*np.sqrt(np.sum(correct[k+1:,k]**2))
    correct_r=np.sqrt(0.5*correct_a*correct_a-0.5*correct_a*correct[k+1][k])
    w = np.zeros(correct.shape[0])
    w[k + 1] = (correct[k + 1, k] - correct_a) / (2 * correct_r)
    w[k + 2:] = correct[k + 2:, k] / (2 * correct_r)
    #p=np.eye(4)-2*np.outer(w,w) calc p
    return w


def run_svd(A:np.ndarray):
    create_tables()
    store_matrices(A)
    compute_summation_matrices()
    calculate_correlation_matrix()
    perform_householder()
    execute_qr()

def display_table(s:str):
    t=np.array(db.sql(f"SELECT * FROM t{s}").fetchall())
    print(t)
    return t
    
if __name__ == "__main__":
    A=np.array([[1,2,3,4],[5,6,7,8],[1,2,3,4]]) 
    run_svd(A)
    display_table("A")
    # q=display_table("H")
    # w=calc_w(A,0)
    # p=np.eye(4)-2*np.outer(w,w)
    # print(np.eye(4)@p)
    # H=np.corrcoef(np.transpose(A))
    # print(p)
    # print(H)
    # print(p@H@p)
    # out=np.reshape(q[:,2],H.shape)
    # print(out)
    # print(np.allclose(out,p@H@p))
    # test_alpha_calc()
    # print(np.corrcoef(np.transpose(A)))
    # display_table('H')
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
    # test_correlation_matrix_calc()