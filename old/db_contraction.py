import duckdb
import sqlite3
from timeit import default_timer as timer
from abc import ABC


class abstractDB(ABC):
    
    def contraction(self,str:str):
        pass
    
    def close(self):
        pass

class duckDB(abstractDB):
    def __init__(self) -> None:
        pass
    
    def contraction(self,str:str):
        res=duckdb.sql(str).fetchall()
        return res
    
    def close(self):
        pass

class sqliteDB(abstractDB):
    def __init__(self) -> None:
        self.conn=sqlite3.connect(':memory:')

    def contraction(self,str:str):
        res=self.conn.cursor().execute(str)
        self.conn.commit()
        return res.fetchall()
    
    def close(self):
        self.conn.close()