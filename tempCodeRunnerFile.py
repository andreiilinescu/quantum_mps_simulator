for idx, v in np.ndenumerate(U):
                  if v !=0.0:
                        self.conn.execute(f"INSERT INTO t{first_qbit} (i, j,k, re,im) VALUES ({idx[0]},{idx[1]},{idx[2]} , {v.real}, {v.imag})")
  