import duckdb
from timeit import default_timer as timer

def duckdb_contraction(str:str):
    tic=timer()
    res=duckdb.sql(str).fetchall()
    toc=timer()
    return res,toc-tic
