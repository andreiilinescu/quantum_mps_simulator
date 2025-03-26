import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import json

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

def plot_multiple_lines(max_qbits:int,ys,labels, x_axis_label:str='X-axis',y_axis_label:str='Y-axis',title:str='Multiple Lines Connecting Scatter Points', min_qbits:int=2,step:int=1):
    x = list(range(min_qbits, max_qbits + 1,step))
    for y, label in zip(ys, labels):
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length. Found {len(x)} and {len(y)} for label '{label}'.")
        plt.scatter(x, y, label=label)  # Plot scatter points
        plt.plot(x, y, linestyle='-', marker='o')  # Connect points with a line and add markers for clarity
    
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.7)  # Improve grid visibility
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

def plot_stacked_bars(xlabels: list[str], avgs: list[dict[str, float]], title: str = 'Time Breakdown by Contraction Type'):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Use matplotlib's default color cycle.
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Dictionary to map each section to a specific color.
    section_color = {}
    color_index = 0

    # Set to track labels already used in the legend.
    seen_labels = set()

    # Helper function to plot a single stacked bar.
    def plot_stacked_bar(ax, x, breakdown):
        nonlocal color_index
        bottom = 0
        for section, frac in breakdown.items():
            # Assign a color to the section if it hasn't been assigned yet.
            if section not in section_color:
                section_color[section] = color_cycle[color_index % len(color_cycle)]
                color_index += 1
            color = section_color[section]
            
            # Only assign the label once for the legend.
            label = section if section not in seen_labels else None
            seen_labels.add(section)
            
            ax.bar(x, frac, bottom=bottom, color=color, label=label)
            bottom += frac

    # Plot each bar.
    for lbl, avg in zip(xlabels, avgs):
        plot_stacked_bar(ax, lbl, avg)

    ax.set_ylabel('Fraction of Total Time')
    ax.set_title(title)
    ax.legend()

    plt.show()



if __name__=="__main__":
    data=json.load(open("./new_data/ghz_PC-2080s_(200_1000_100)_100.json"))
    data2=json.load(open("./new_data/qiskit/ghz_PC-2080s_(200_1000_100)_100.json"))
    plot_multiple_lines(data["max_qubits"],[list(data["times"].values()),list(data2["times"].values())],["sqlite_mps","qiskit_aer_mps"],"Number of Qubits","Time (s)","Sqlite MPS vs Qiskit Aer MPS",data["min_qubits"],data["step_qubits"])