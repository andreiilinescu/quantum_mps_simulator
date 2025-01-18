import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def plot_statevector(MPS: np.ndarray):
    MPS=MPS.flatten()
    n = int(np.log2(MPS.shape[0]))
    probs = np.abs(MPS).__pow__(2)

    x_lbls = ["".join(state) for state in product("01", repeat=n)]

    plt.bar(x_lbls, probs)
    plt.xlabel("Quantum States")
    plt.ylabel("Probability")
    plt.title("Probability Distribution of  Quantum States")
    plt.ylim(0, 1)
    plt.show()

def plot_multiple_lines(num_qubits,ys,labels, x_axis_label:str='X-axis',y_axis_label:str='Y-axis',title:str='Multiple Lines Connecting Scatter Points'):
    if len(ys) != len(labels):
        raise ValueError("The number of lines (rows in ys) must match the number of labels.")
    x=list(range(2,num_qubits+1))
    for y, label in zip(ys, labels):
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length. Found {len(x)} and {len(y)} for label '{label}'.")
        plt.scatter(x, y, label=label)  # Plot scatter points
        plt.plot(x, y)                 # Connect points with a line
    
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


