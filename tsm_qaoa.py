import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo


# ============================================================
# 1) Piccola istanza TSP: 4 città -> 4^2 = 16 qubit logici
# ============================================================
#dist = np.array(
#    [
#        [0, 10, 15, 20],
#        [10, 0, 35, 25],
#        [15, 35, 0, 30],
#        [20, 25, 30, 0],
#    ],
#    dtype=float,
#)
print("hello")

dist = np.array([
    [0.0, 1.0, 1.41421356, 1.0],
    [1.0, 0.0, 1.0, 1.41421356],
    [1.41421356, 1.0, 0.0, 1.0],
    [1.0, 1.41421356, 1.0, 0.0],
])

'''
dist = np.array([
    [0.0,        1.1755705,  1.90211303, 1.90211303, 1.1755705 ],
    [1.1755705,  0.0,        1.1755705,  1.90211303, 1.90211303],
    [1.90211303, 1.1755705,  0.0,        1.1755705,  1.90211303],
    [1.90211303, 1.90211303, 1.1755705,  0.0,        1.1755705 ],
    [1.1755705,  1.90211303, 1.90211303, 1.1755705,  0.0       ]
])
'''
n = len(dist)

# Tsp accetta anche una matrice di adiacenza / distanze
tsp = Tsp(dist)
qp = tsp.to_quadratic_program()

qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qp)

cost_hamiltonian, offset = qubo.to_ising()

print(qp.prettyprint())
print("\nNumero di qubit logici:", cost_hamiltonian.num_qubits)
print("Offset:", offset)


# ============================================================
# 2) Circuito QAOA
# ============================================================
reps = 4

circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

print("\nParametri del circuito:")
print(circuit.parameters)

# Per QAOAAnsatz, l'ordine tipico è beta..., gamma...
initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [initial_beta] * reps + [initial_gamma] * reps


# ============================================================
# 3) Funzione costo con StatevectorEstimator
# ============================================================
objective_func_vals = []
estimator = StatevectorEstimator(seed=123)


def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])
    result = job.result()[0]

    # result.data.evs può essere scalar-like o array shape ()
    cost = float(np.asarray(result.data.evs).item())

    objective_func_vals.append(cost)
    return cost


# ============================================================
# 4) Ottimizzazione classica dei parametri
# ============================================================
result = minimize(
    cost_func_estimator,
    init_params,
    args=(circuit, cost_hamiltonian, estimator),
    method="COBYLA",
    tol=1e-2,
    options={"maxiter": 50},
)

print("\nRisultato ottimizzazione:")
print(result)


# ============================================================
# 5) Plot convergenza costo
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("QAOA optimization (statevector simulation)")
plt.show()


# ============================================================
# 6) Campionamento finale con StatevectorSampler
# ============================================================
optimized_circuit = circuit.assign_parameters(result.x)
optimized_circuit.measure_all()

sampler = StatevectorSampler(seed=123)
job = sampler.run([(optimized_circuit,)], shots=10000)
sample_result = job.result()[0]

# Con measure_all() il registro classico si chiama di solito "meas"
counts_bin = sample_result.data.meas.get_counts()
shots = sum(counts_bin.values())
final_distribution_bin = {k: v / shots for k, v in counts_bin.items()}


# ============================================================
# 7) Helper per convertire bitstring/counts in soluzione TSP
# ============================================================
def bitstring_to_little_endian_array(bitstring: str) -> np.ndarray:
    """
    Qiskit mostra le stringhe in ordine big-endian per la stampa.
    Per avere x[0] associato al qubit 0, invertiamo.
    """
    return np.array([int(b) for b in bitstring[::-1]], dtype=int)


def decode_candidate(bitstring: str):
    x_qubo = bitstring_to_little_endian_array(bitstring)
    x_original = qubo_converter.interpret(x_qubo)
    feasible = qp.is_feasible(x_original)
    route = tsp.interpret(x_original)

    length = None
    if feasible and isinstance(route, list) and len(route) == n:
        if all(isinstance(v, (int, np.integer)) for v in route):
            length = Tsp.tsp_value(route, dist)

    return {
        "bitstring": bitstring,
        "x_qubo": x_qubo,
        "x_original": x_original,
        "feasible": feasible,
        "route": route,
        "length": length,
    }


# ============================================================
# 8) Candidato più probabile
# ============================================================
most_likely_bitstring = max(final_distribution_bin, key=final_distribution_bin.get)
decoded_most_likely = decode_candidate(most_likely_bitstring)

print("\nBitstring più probabile:")
print(most_likely_bitstring)
print("Feasible:", decoded_most_likely["feasible"])
print("Tour decodificato:", decoded_most_likely["route"])
print("Lunghezza tour:", decoded_most_likely["length"])


# ============================================================
# 9) Miglior soluzione ammissibile tra i campioni
#    (più robusto del solo 'most likely')
# ============================================================
best_feasible = None

for bitstring, prob in final_distribution_bin.items():
    decoded = decode_candidate(bitstring)
    if decoded["feasible"] and decoded["length"] is not None:
        if best_feasible is None or decoded["length"] < best_feasible["length"]:
            best_feasible = decoded | {"probability": prob}

print("\nMiglior soluzione ammissibile trovata nei campioni:")
if best_feasible is None:
    print("Nessuna soluzione ammissibile trovata nei campioni.")
else:
    print("Bitstring:", best_feasible["bitstring"])
    print("Probabilità:", best_feasible["probability"])
    print("Tour:", best_feasible["route"])
    print("Lunghezza tour:", best_feasible["length"])


# ============================================================
# 10) Plot delle 10 stringhe più probabili
# ============================================================
top_items = sorted(final_distribution_bin.items(), key=lambda kv: kv[1], reverse=True)[:10]

plt.figure(figsize=(12, 5))
plt.bar([k for k, _ in top_items], [v for _, v in top_items])
plt.xticks(rotation=45)
plt.xlabel("Bitstrings")
plt.ylabel("Probability")
plt.title("Top sampled bitstrings (statevector sampler)")
plt.show()