#%%
import matplotlib
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from typing import Sequence

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

import heapq
import math
from tsp_planner import (
    plan_tsp_instance,
    plot_feasible_graph,
    plot_best_tsp_subset,
)
import heapq
import math
from collections import defaultdict
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

places = {
    1: ("Pico Aneto", "Peak", 3404),
    2: ("Tuca Alba", "Peak", 3122),
    3: ("Benasque", "Town", 1135),
    4: ("Cerler", "Town", 1530),
    5: ("Ancils", "Town", 1140),
    6: ("Portillón de Benás", "Landmark", 2444),
    7: ("Ski resort", "Snow", 1530),
    8: ("Baños de Benasque", "Resting area", 1550),
    9: ("Hospital de Benasque", "Resting area", 1747),
    10: ("Forau d.aiguallut", "Landmark", 2020),
    11: ("Tres Cascadas", "Landmark", 1900),
    12: ("Salvaguardia", "Peak", 2736),
    13: ("Tuca Maladeta", "Peak", 3312),
    14: ("Cap Llauset", "Refugio", 2425),
    15: ("Ibón Cregüeña", "Lake", 2632),
    16: ("Batisielles", "Lake", 2216),
    17: ("Eriste", "Town", 1089),
    18: ("Ibón Eriste", "Lake", 2407),
    19: ("Tempestades", "Peak", 3289),
    20: ("La Besurta", "Resting area", 1860),
    21: ("La Renclusa", "Refugio", 2160),
    22: ("Escaleta", "Lake", 2630),
    23: ("Mulleres", "Peak", 3013),
    24: ("Salterillo", "Lake", 2460),
    25: ("Tres Barrancos", "Landmark", 1460),
}

edges = [
    (1, 21, "4'50"),
    (1, 25, "4'35"),

    (2, 8, "4'20"),

    (3, 4, "1'25"),
    (3, 5, "0'45"),
    (3, 8, "0"),
    (3, 9, "0"),
    (3, 16, "3'10"),
    (3, 17, "0"),
    (3, 20, "0"),
    (3, 25, "1'00"),

    (4, 5, "1'30"),
    (4, 7, "0"),
    (4, 8, "0"),
    (4, 9, "0"),
    (4, 11, "1'10"),
    (4, 17, "0"),
    (4, 20, "0"),

    (5, 8, "0"),
    (5, 9, "0"),
    (5, 17, "0"),
    (5, 20, "0"),

    (6, 9, "2'25"),
    (6, 12, "1'00"),

    (8, 15, "3'35"),
    (8, 20, "0"),

    (9, 20, "0"),

    (10, 21, "1'00"),
    (10, 22, "1'50"),

    (13, 20, "4'20"),

    (14, 19, "3'40"),
    (14, 21, "5'50"),

    (16, 18, "2'30"),

    (20, 21, "4'20"),
    (20, 22, "3'05"),

    (21, 24, "1'35"),

    (22, 23, "1'15"),

    (23, 24, "0'50"),
    ]


REPS = 5

result = plan_tsp_instance(
    season="summer",
    expertise="intermediate",
    n_sites=6,
    base_node=3,
    preferred_types=["Peak"],
    require_preferred=False,
    preference_bonus=60,
    max_town_like=2,
    min_degree=2,
    use_time_filter=True,
    altitude_factor=0.05,
    zero_replacement=5
)

print("Best nodes:", result["best_nodes"])
print("Weight matrix shape:", None if result["weight_matrix"] is None else result["weight_matrix"].shape)
print("Weight matrix:")
print(result["weight_matrix"])

print("Labels:")
print(result["matrix_labels"])

plot_feasible_graph(result["G_selected"], base_node=3)

if result["best_nodes"] is not None:
    plot_best_tsp_subset(
        G_selected=result["G_selected"],
        selected_df=result["selected_places_df"],
        best_nodes=result["best_nodes"],
        base_node=3
    )

# ------------------------------------------------------------
# 1) Use the planner's results
# ------------------------------------------------------------
tsp_nodes = result["best_nodes"]                # e.g. [3, 20, 22, 23, 25]
matrix_labels = result["matrix_labels"]          # e.g. ['3 - Benasque', '20 - La Besurta', ...]
weight_matrix = result["weight_matrix"]          # numpy array, shape (N, N)

N = len(tsp_nodes)
print("TSP nodes selected:", tsp_nodes)

# Build mapping from node ID to matrix index by parsing the label
node_to_mat_idx = {}
for idx, label in enumerate(matrix_labels):
    # Extract the node number: everything before the first space or dash
    node_id = int(label.split()[0])              # e.g. '3' from '3 - Benasque'
    node_to_mat_idx[node_id] = idx

# Create a distance dictionary for quick lookup
dist = {}
for i_node in tsp_nodes:
    for j_node in tsp_nodes:
        i_idx = node_to_mat_idx[i_node]
        j_idx = node_to_mat_idx[j_node]
        dist[(i_node, j_node)] = float(weight_matrix[i_idx, j_idx])

# ------------------------------------------------------------
# 2) Binary encoding setup (same as before)
# ------------------------------------------------------------
k = math.ceil(math.log2(N)) if N > 1 else 1
n_qubits_tsp = N * k
print(f"Binary encoding: N={N} cities, k={k} bits per city → {n_qubits_tsp} total qubits")
print(f"(vs. {N*N} qubits for one-hot encoding)")

def pos_bits(city_idx, k):
    """Return qubit indices for binary position of city_idx."""
    return [city_idx * k + bit for bit in range(k)]

# Map city ID (the original node number) to an index 0..N-1 for our internal ordering
node_to_idx = {node: idx for idx, node in enumerate(tsp_nodes)}
idx_to_node = {idx: node for node, idx in node_to_idx.items()}

# ------------------------------------------------------------
# 3) Build QUBO (identical logic, only the distance lookup changes)
# ------------------------------------------------------------
A = 10.0   # constraint penalty
B = 1.0    # distance weight

qubo_lin = defaultdict(float)
qubo_quad = defaultdict(float)
qubo_const = 0.0

def add_lin(i, c):
    qubo_lin[i] += c

def add_quad(i, j, c):
    if i == j:
        qubo_lin[i] += c
    elif i < j:
        qubo_quad[(i, j)] += c
    else:
        qubo_quad[(j, i)] += c

# Constraint: penalize duplicate positions (all cities must have distinct positions)
for i in range(N):
    for j in range(i + 1, N):
        bits_i = pos_bits(i, k)
        bits_j = pos_bits(j, k)
        for b in range(k):
            add_lin(bits_i[b], A)
            add_lin(bits_j[b], A)
            add_quad(bits_i[b], bits_j[b], -2.0 * A)

# Objective: minimise cycle distance
# For each consecutive pair in the tour, penalise based on their distance
for city_order in range(N):
    ni = tsp_nodes[city_order]               # actual node ID of the city at this position
    nj = tsp_nodes[(city_order + 1) % N]
    i = node_to_idx[ni]                      # internal index for city i
    j = node_to_idx[nj]                      # internal index for city j

    for pos_i in range(N):
        for pos_j in range(N):
            # Use the precomputed distance from the dictionary
            dist_val = dist[(ni, nj)]
            weight = B * dist_val / (N * N)

            bits_i = pos_bits(i, k)
            bits_j = pos_bits(j, k)

            for b in range(k):
                if ((pos_i >> b) & 1) == 1:
                    add_lin(bits_i[b], weight)
                else:
                    add_lin(bits_i[b], -weight)

                if ((pos_j >> b) & 1) == 1:
                    add_lin(bits_j[b], weight)
                else:
                    add_lin(bits_j[b], -weight)

# ------------------------------------------------------------
# 4) QUBO → Ising → SparsePauliOp (identical)
# ------------------------------------------------------------
z_lin = defaultdict(float)
zz = defaultdict(float)
energy_offset = qubo_const

for i, c in qubo_lin.items():
    energy_offset += c / 2.0
    z_lin[i] += -c / 2.0

for (i, j), c in qubo_quad.items():
    energy_offset += c / 4.0
    z_lin[i] += -c / 4.0
    z_lin[j] += -c / 4.0
    zz[(i, j)] += c / 4.0

pauli_terms_tsp = []
for i, c in z_lin.items():
    if abs(c) > 1e-12:
        pauli_terms_tsp.append(("Z", [i], c))
for (i, j), c in zz.items():
    if abs(c) > 1e-12:
        pauli_terms_tsp.append(("ZZ", [i, j], c))

tsp_cost_hamiltonian = SparsePauliOp.from_sparse_list(pauli_terms_tsp, n_qubits_tsp).simplify()
print(f"Hamiltonian terms: {len(tsp_cost_hamiltonian.paulis)}")

# ------------------------------------------------------------
# 5) QAOA circuit (identical)
# ------------------------------------------------------------
circuit_tsp = QAOAAnsatz(cost_operator=tsp_cost_hamiltonian, reps=REPS)
circuit_tsp.measure_all()


from qiskit import transpile
from qiskit_aer import AerSimulator

# =========================================================
# Run QAOA (local simulator) and decode best TSP path
# =========================================================

sim = AerSimulator()

# Build a no-measure circuit template (reps=1 => 2 parameters)
qaoa_template = QAOAAnsatz(cost_operator=tsp_cost_hamiltonian, reps=REPS)
qaoa_template_t = transpile(qaoa_template, sim)


def decode_bitstring_to_tour_binary(bitstring: str, N: int, k: int):
    """
    Decode binary-encoded position: each city i occupies k bits.
    Read those bits as binary number to get its tour position.
    """
    bits = bitstring[::-1]  # qubit index order
    
    positions = []
    for city_idx in range(N):
        # Extract k bits for this city
        city_bits = pos_bits(city_idx, k)
        
        # Convert to position value (binary interpretation)
        pos_value = 0
        for b_idx, qubit_idx in enumerate(city_bits):
            if qubit_idx < len(bits):
                pos_value += int(bits[qubit_idx]) * (2 ** b_idx)
        
        # Clamp to valid range [0, N-1]
        pos_value = min(pos_value, N - 1)
        positions.append((pos_value, city_idx))
    
    # Sort by position to get tour order
    positions.sort()
    tour_idx = [city_idx for pos, city_idx in positions]
    
    return tour_idx

def tour_cost_nodes(tour_nodes, dist_map):
    total = 0.0
    for p in range(len(tour_nodes)):
        a = tour_nodes[p]
        b = tour_nodes[(p + 1) % len(tour_nodes)]
        total += dist_map[(a, b)]
    return total


from scipy.optimize import minimize

# --- Cost function for the optimizer ---
def qaoa_cost(params):
    """
    params: list of 8 numbers (4 gammas + 4 betas)
    returns: total distance of the best tour found for these parameters
    """
    bound = qaoa_template_t.assign_parameters(params)
    qc = bound.copy()
    qc.measure_all()
    result = sim.run(qc, shots=256).result()          # more shots = better estimate
    counts = result.get_counts()
    best_bit = max(counts, key=counts.get)
    tour_idx = decode_bitstring_to_tour_binary(best_bit, N, k)
    tour_nodes = [idx_to_node[i] for i in tour_idx]
    return tour_cost_nodes(tour_nodes, dist)

# --- Initial guess ---
# If you already ran the coarse grid with reps=1, use its best values:
# best_gamma = best["gamma"]   (from your earlier result)
# best_beta  = best["beta"]
# Otherwise, use a sensible default, e.g.:
best_gamma = 1.0
best_beta  = 0.5

# Repeat for all 4 layers (so 8 parameters total)
x0 = [best_gamma, best_beta] * REPS  # length 8

# --- Run the optimisation ---
print("Starting optimisation with 4 QAOA layers...")
res = minimize(qaoa_cost, x0, method='COBYLA', options={'maxiter': 200})
print("Optimised parameters:", res.x)

# --- Evaluate the final best tour ---
final_cost = qaoa_cost(res.x)
print("\nFinal best tour cost:", final_cost)

# Decode the tour from the optimised parameters for plotting
bound = qaoa_template_t.assign_parameters(res.x)
qc = bound.copy()
qc.measure_all()
result = sim.run(qc, shots=1024).result()
counts = result.get_counts()
best_bit = max(counts, key=counts.get)
tour_idx = decode_bitstring_to_tour_binary(best_bit, N, k)
tour_nodes = [idx_to_node[i] for i in tour_idx]
print("Best tour:", tour_nodes + [tour_nodes[0]])   # close the cycle

# -----------------------------
# Plot best tour over selected nodes
# -----------------------------
selected_nodes = tsp_nodes   # this is the set of nodes used for TSP

# Build an rx complete graph just to compute a stable spring-layout x-coordinate
H = rx.PyGraph()
for n in selected_nodes:
    H.add_node(n)
for i in range(len(selected_nodes)):
    for j in range(i + 1, len(selected_nodes)):
        u = selected_nodes[i]
        v = selected_nodes[j]
        H.add_edge(i, j, dist[(u, v)])

base_pos = rx.spring_layout(H, seed=42)
pos_tour = {
    selected_nodes[idx]: (base_pos[idx][0], places[selected_nodes[idx]][2])
    for idx in range(len(selected_nodes))
}

# Use the tour found by the optimiser (already in variable tour_nodes)
route_edges = [(tour_nodes[i], tour_nodes[(i + 1) % len(tour_nodes)]) for i in range(len(tour_nodes))]
route_edges_undirected = {tuple(sorted(e)) for e in route_edges}

plt.figure(figsize=(9, 6))

# Draw all pair edges (metric closure), highlighting route edges
for i in range(len(selected_nodes)):
    for j in range(i + 1, len(selected_nodes)):
        u = selected_nodes[i]
        v = selected_nodes[j]
        x = [pos_tour[u][0], pos_tour[v][0]]
        y = [pos_tour[u][1], pos_tour[v][1]]
        if tuple(sorted((u, v))) in route_edges_undirected:
            plt.plot(x, y, linewidth=3.2, color="#e63946", zorder=2)
        else:
            plt.plot(x, y, linewidth=1.2, color="lightgray", alpha=0.8, zorder=1)

        mx = 0.5 * (x[0] + x[1])
        my = 0.5 * (y[0] + y[1])
        plt.text(mx, my, f"{dist[(u, v)]:.2f}", fontsize=7, color="dimgray")

# Draw nodes and labels
xs = [pos_tour[n][0] for n in selected_nodes]
ys = [pos_tour[n][1] for n in selected_nodes]
plt.scatter(xs, ys, s=750, c="#d9ecff", edgecolors="black", zorder=3)

for n in selected_nodes:
    plt.text(
        pos_tour[n][0],
        pos_tour[n][1],
        f"{n}: {places[n][0]}",
        fontsize=8,
        ha="center",
        va="center",
        zorder=4,
    )

plt.ylabel("Altitude")
plt.xticks([])
plt.grid(axis="y", linestyle=":", alpha=0.35)
plt.tight_layout()
plt.show()

# #%%

# # Fast coarse parameter search
# gammas = np.linspace(0.0, np.pi, 4)
# betas = np.linspace(0.0, np.pi / 2, 4)

# best = {
#     "cost": float("inf"),
#     "gamma": None,
#     "beta": None,
#     "tour_nodes": None,
#     "bitstring": None,
# }

# shots = 1024

# from scipy.optimize import minimize

# # --- Cost function for the optimizer ---
# def qaoa_cost(params):
#     # params is a flat list of length 2*reps
#     bound = qaoa_template_t.assign_parameters(params)
#     qc = bound.copy()
#     qc.measure_all()
#     result = sim.run(qc, shots=256).result()          # more shots = better estimate
#     counts = result.get_counts()
#     best_bit = max(counts, key=counts.get)
#     tour_idx = decode_bitstring_to_tour_binary(best_bit, N, k)
#     tour_nodes = [idx_to_node[i] for i in tour_idx]
#     return tour_cost_nodes(tour_nodes, dist)          # minimise actual tour distance

# # --- Initial guess from your coarse grid (reps=1) ---
# # Assume best["gamma"] and best["beta"] are the best values from the 1‑layer search
# x0 = [best["gamma"], best["beta"]] * reps   # repeat for each layer

# # --- Run optimisation ---
# res = minimize(qaoa_cost, x0, method='COBYLA', options={'maxiter': 200})
# print("Optimised parameters:", res.x)

# # --- Evaluate final tour ---
# final_cost = qaoa_cost(res.x)   # or call decode again manually

# for gamma in gammas:
#     for beta in betas:
#         bound = qaoa_template_t.assign_parameters([gamma, beta])
#         qc = bound.copy()
#         qc.measure_all()

#         result = sim.run(qc, shots=shots).result()
#         counts = result.get_counts()

#         best_bit = max(counts, key=counts.get)
#         tour_idx = decode_bitstring_to_tour_binary(best_bit, N, k)
#         tour_nodes = [idx_to_node[i] for i in tour_idx]
#         c = tour_cost_nodes(tour_nodes, dist)

#         if c < best["cost"]:
#             best.update(
#                 {
#                     "cost": c,
#                     "gamma": float(gamma),
#                     "beta": float(beta),
#                     "tour_nodes": tour_nodes,
#                     "bitstring": best_bit,
#                 }
#             )

# print("Best QAOA parameters (binary encoding, coarse grid)")
# print("gamma:", best["gamma"], "beta:", best["beta"])
# print("Best decoded tour:", best["tour_nodes"] + [best["tour_nodes"][0]])
# print("Tour cost:", best["cost"])

#%%
# -----------------------------
# Plot best tour over selected 5-node metric graph (without networkx)
# -----------------------------
selected_nodes = tsp_nodes

# Build an rx complete graph just to compute a stable spring-layout x-coordinate
H = rx.PyGraph()
for n in selected_nodes:
    H.add_node(n)
for i in range(len(selected_nodes)):
    for j in range(i + 1, len(selected_nodes)):
        u = selected_nodes[i]
        v = selected_nodes[j]
        H.add_edge(i, j, dist[(u, v)])

base_pos = rx.spring_layout(H, seed=42)
pos_tour = {
    selected_nodes[idx]: (base_pos[idx][0], places[selected_nodes[idx]][2])
    for idx in range(len(selected_nodes))
}

tour_nodes = best["tour_nodes"]
route_edges = [(tour_nodes[i], tour_nodes[(i + 1) % len(tour_nodes)]) for i in range(len(tour_nodes))]
route_edges_undirected = {tuple(sorted(e)) for e in route_edges}

plt.figure(figsize=(9, 6))

# Draw all pair edges (metric closure), highlighting route edges
for i in range(len(selected_nodes)):
    for j in range(i + 1, len(selected_nodes)):
        u = selected_nodes[i]
        v = selected_nodes[j]
        x = [pos_tour[u][0], pos_tour[v][0]]
        y = [pos_tour[u][1], pos_tour[v][1]]
        if tuple(sorted((u, v))) in route_edges_undirected:
            plt.plot(x, y, linewidth=3.2, color="#e63946", zorder=2)
        else:
            plt.plot(x, y, linewidth=1.2, color="lightgray", alpha=0.8, zorder=1)

        mx = 0.5 * (x[0] + x[1])
        my = 0.5 * (y[0] + y[1])
        plt.text(mx, my, f"{dist[(u, v)]:.2f}", fontsize=7, color="dimgray")

# Draw nodes and labels
xs = [pos_tour[n][0] for n in selected_nodes]
ys = [pos_tour[n][1] for n in selected_nodes]
plt.scatter(xs, ys, s=750, c="#d9ecff", edgecolors="black", zorder=3)

for n in selected_nodes:
    plt.text(
        pos_tour[n][0],
        pos_tour[n][1],
        f"{n}: {places[n][0]}",
        fontsize=8,
        ha="center",
        va="center",
        zorder=4,
    )

plt.ylabel("Altitude")
plt.xticks([])
plt.grid(axis="y", linestyle=":", alpha=0.35)
plt.tight_layout()
plt.show()

#%%