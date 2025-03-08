import numpy as np
import matplotlib.pyplot as plt
from sqlite_mps_full_analytics import SQLITE_MPS
from performance import generate_ghz_circuit
from plotting import plot_stacked_bars
KEYS_ONE = ['pure_contraction', 'cleanup']
KEYS_TWO = ['pure_contraction', 'svd', 'cleanup']

def compute_average_breakdown(group, keys):
    """
    For each timing dictionary in the group, compute the fraction of total_time
    for each key in 'keys'. Then return the average fraction for each key.
    """
    fractions = {k: [] for k in keys}
    for d in group:
        total = d['total_time']
        for k in keys:
            # If the key is missing (e.g. 'svd' for 1-qubit), treat it as 0.
            fractions[k].append(d.get(k, 0) / total)

    return {k: np.mean(v) for k, v in fractions.items()}


def simulate_median_breakdown(num_qbits:int ,iters:int):
    circuit=generate_ghz_circuit(num_qbits)
    one_qubit=[]
    two_qubit=[]
    for i in range(iters):
        sim=SQLITE_MPS.run_circuit_json(circuit)
        times=sim.times
        # Separate timing entries into 1-qubit and 2-qubit groups
        one_qubit = one_qubit+[d for d in times if 'svd' not in d]
        two_qubit = two_qubit+[d for d in times if 'svd' in d]
        
    avg_one = compute_average_breakdown(one_qubit, KEYS_ONE)
    avg_two = compute_average_breakdown(two_qubit, KEYS_TWO)
    return avg_one,avg_two

# Define the keys that should be present in each group.
# For one-qubit contractions, we only have 'pure_contraction' and 'cleanup'
# For two-qubit contractions, we have an extra 'svd' component.

# Compute average breakdown fractions for each group.


avg_one,avg_two=simulate_median_breakdown(10,100)
print(avg_one)
# Now create a stacked bar chart.
labels = ['1-qubit', '2-qubit']

plot_stacked_bars(labels,[avg_one,avg_two])




