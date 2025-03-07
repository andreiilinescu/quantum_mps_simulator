from sqlite_mps_full_analytics import SQLITE_MPS_Analytics
from timeit import default_timer as timer
import numpy as np
import json
from plotting import plot_multiple_lines

def generate_ghz_circuit(num_qubits: int):
    """Generates a GHZ circuit JSON description for the given number of qubits."""
    data = {"number_of_qubits": num_qubits, "gates": []}
    data["gates"].append({"qubits": [0], "gate": "H"})
    for i in range(num_qubits - 1):
        data["gates"].append({"qubits": [i, i + 1], "gate": "CNOT"})
    return data

def aggregate_run_metrics(analytics_list):
    """
    Aggregates a single run's analytics by summing all numeric metrics from each gate.
    Non-numeric keys (such as gate name or memory tuple) are skipped.
    """
    aggregated = {}
    for gate in analytics_list:
        for key, value in gate.items():
            if isinstance(value, (int, float)):
                aggregated[key] = aggregated.get(key, 0) + value
    return aggregated

def analytics_ghz_execution_full(max_qubit: int, num_iterations: int):
    """
    Runs the GHZ circuit for qubit counts from 2 to max_qubit.
    For each circuit size and for each iteration, aggregates per-gate analytics into a single dictionary.
    
    Returns a dictionary with the raw aggregated analytics per run.
    """
    data = {
        "name": "SQLITE_MPS GHZ Full Analytics",
        "max_qubits": max_qubit,
        "num_iterations": num_iterations,
        "analytics": {}  # keys will be qubit count as strings, values are lists of aggregated metric dicts
    }
    
    for q in range(2, max_qubit + 1):
        circ = generate_ghz_circuit(q)
        data["analytics"][str(q)] = []
        for i in range(num_iterations):
            # Run circuit with full analytics
            s = SQLITE_MPS_Analytics.run_circuit_json(circ)
            # Aggregate all numeric metrics (e.g., total_time, cleanup, pure_contraction, svd, etc.)
            agg_metrics = aggregate_run_metrics(s.analytics)
            data["analytics"][str(q)].append(agg_metrics)
    return data

def compute_full_statistics(analytics_data: dict):
    """
    For each qubit count, computes statistics (mean, median, max) for each aggregated metric
    across all iterations.
    
    Returns a nested dictionary structured as:
      { "<qubit_count>": { "<metric>": {"mean": ..., "median": ..., "max": ...}, ... }, ... }
    """
    stats = {}
    for qubit_str, runs in analytics_data["analytics"].items():
        # Determine the union of all metric keys that appeared in any run.
        keys = set()
        for run in runs:
            keys.update(run.keys())
        
        stats[qubit_str] = {}
        for key in keys:
            # For each key, get a list of values over the runs (if the key is present)
            values = [run[key] for run in runs if key in run]
            if values:
                stats[qubit_str][key] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "max": float(np.max(values))
                }
    return stats

def save_data_to_file(data, filename: str):
    """Saves the provided data as JSON into the ./data directory."""
    if not filename.endswith(".json"):
        filename += ".json"
    with open("./data/" + filename, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    # Set parameters – here we run for up to 1000 qubits with a given number of iterations.
    max_qubit = 10
    num_iterations = 100  # Adjust this for better statistics (more iterations = longer runtime)
    
    # Run full analytics on GHZ circuits.
    full_analytics_data = analytics_ghz_execution_full(max_qubit, num_iterations)
    
    # Compute per-qubit statistics: mean, median, and max for each metric.
    full_stats = compute_full_statistics(full_analytics_data)
    
    # (Optional) Plot one or more metrics.
    # For example, to plot the median total time vs. number of qubits:
    qubits = list(range(2, max_qubit + 1))
    median_total_time = [full_stats[str(q)]["total_time"]["median"] for q in qubits if "total_time" in full_stats[str(q)]]
    
    plot_multiple_lines(
        max_qubit,
        [median_total_time],
        ["Median Total Time"],
        "Number of Qubits",
        "Time (s)",
        "GHZ Circuit: Median Total Time (up to 1000 qubits)"
    )
    
    # Save the computed statistics to a file.
    save_data_to_file(full_stats, "ghz_full_stats_10")
