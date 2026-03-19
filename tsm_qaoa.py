import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator

from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit_quantuminspire.qi_provider import QIProvider


# ============================================================
# 1) Problem instance
# ============================================================
print("hello")

dist = np.array([
    [0.0, 1.0, 1.41421356],
    [1.0, 0.0, 1.0],
    [1.41421356, 1.0, 0.0],
], dtype=float)

'''
dist = np.array([
    [0.0, 1.0, 1.41421356, 1.0],
    [1.0, 0.0, 1.0, 1.41421356],
    [1.41421356, 1.0, 0.0, 1.0],
    [1.0, 1.41421356, 1.0, 0.0],
])
'''
n = len(dist)

tsp = Tsp(dist)
qp = tsp.to_quadratic_program()

# Optional: fix city 0 as the first city
# qp.linear_constraint(
#     linear={"x_0_0": 1},
#     sense="==",
#     rhs=1,
#     name="fix_start_city",
# )

qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qp)
cost_hamiltonian, offset = qubo.to_ising()

print(qp.prettyprint())
print("\nNumero di qubit logici:", cost_hamiltonian.num_qubits)
print("Offset:", offset)


# ============================================================
# 2) QAOA circuit
# ============================================================
reps = 3   # 10 è molto alto per un test così piccolo
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

print("\nParametri del circuito:")
print(circuit.parameters)

initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [initial_beta] * reps + [initial_gamma] * reps


# ============================================================
# 3) Cost function with local statevector estimator
# ============================================================
objective_func_vals = []
estimator = StatevectorEstimator(seed=123)


def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])
    result = job.result()[0]
    cost = float(np.asarray(result.data.evs).item())
    objective_func_vals.append(cost)
    return cost


# ============================================================
# 4) Classical optimization
# ============================================================
opt_result = minimize(
    cost_func_estimator,
    init_params,
    args=(circuit, cost_hamiltonian, estimator),
    method="COBYLA",
    tol=1e-2,
    options={"maxiter": 50},
)

print("\nRisultato ottimizzazione:")
print(opt_result)


# ============================================================
# 5) Plot optimization history
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("QAOA optimization (local statevector)")
plt.savefig("qaoa_qi.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 6) Final circuit for Quantum Inspire
# ============================================================
optimized_circuit = circuit.assign_parameters(opt_result.x)
optimized_circuit.measure_all()

provider = QIProvider()
print(provider.backends())

backend_name = "Tuna-9"
backend = provider.get_backend(name=backend_name)
print(backend)

# ------------------------------------------------------------
# IMPORTANT:
# QAOAAnsatz is still high-level. Decompose it before transpiling.
# ------------------------------------------------------------
candidate_circuit = optimized_circuit

for _ in range(8):
    ops = candidate_circuit.count_ops()
    if "QAOA" not in ops and "PauliEvolution" not in ops:
        break
    candidate_circuit = candidate_circuit.decompose()

print("\nOperazioni dopo decompose:")
print(candidate_circuit.count_ops())

# Transpile to backend-supported instructions
candidate_circuit = transpile(
    candidate_circuit,
    backend=backend,
    optimization_level=3,
)

print("\nOperazioni dopo transpile:")
print(candidate_circuit.count_ops())

nr_shots = min(4096, getattr(backend, "max_shots", 4096))
job = backend.run(candidate_circuit, shots=nr_shots)
backend_result = job.result(timeout=600)

counts_bin_raw = backend_result.get_counts()

# Normalize counts in case bitstrings contain spaces
counts_bin = {k.replace(" ", ""): v for k, v in counts_bin_raw.items()}

shots = sum(counts_bin.values())
final_distribution_bin = {k: v / shots for k, v in counts_bin.items()}


# ============================================================
# 7) Helpers
# ============================================================
def bitstring_to_little_endian_array(bitstring: str) -> np.ndarray:
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
# 8) Most likely sampled bitstring
# ============================================================
most_likely_bitstring = max(final_distribution_bin, key=final_distribution_bin.get)
decoded_most_likely = decode_candidate(most_likely_bitstring)

print("\nBitstring più probabile:")
print(most_likely_bitstring)
print("Feasible:", decoded_most_likely["feasible"])
print("Tour decodificato:", decoded_most_likely["route"])
print("Lunghezza tour:", decoded_most_likely["length"])


# ============================================================
# 9) Best feasible solution among samples
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
# 10) Plot top sampled bitstrings
# ============================================================
top_items = sorted(final_distribution_bin.items(), key=lambda kv: kv[1], reverse=True)[:10]

plt.figure(figsize=(12, 5))
plt.bar([k for k, _ in top_items], [v for _, v in top_items])
plt.xticks(rotation=45)
plt.xlabel("Bitstrings")
plt.ylabel("Probability")
plt.title("Top sampled bitstrings (Quantum Inspire)")
plt.savefig("bitstring_qi.png", dpi=300, bbox_inches="tight")
plt.show()