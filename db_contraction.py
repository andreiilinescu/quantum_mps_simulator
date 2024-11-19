import duckdb
from timeit import default_timer as timer

def duckdb_contraction(str:str):
    res=duckdb.sql(str).fetchall()
    return res
