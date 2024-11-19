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

