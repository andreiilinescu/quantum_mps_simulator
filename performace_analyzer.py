from mpsSimulator import SimMPS
def plot_time_diff(nr_iter, nr_qbits, gates):
    sql_times=[]
    np_times=[]
    for i in range(nr_iter):
       t=SimMPS.run(nr_qbits,gates).get_times()
       sql_times.append(t['sql']['total'])
       np_times.append(t['np']['total'])
    print(sql_times)


if __name__ == "__main__":
    plot_time_diff(10,2,[('h',0),('cnot',0,1)])