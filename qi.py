import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, transpile
from qiskit_quantuminspire.qi_provider import QIProvider
from qi_utilities.utility_functions.circuit_modifiers import apply_readout_circuit
from qi_utilities.utility_functions.raw_data_processing import get_multi_counts, get_multi_probs
from qi_utilities.utility_functions.readout_correction import (split_raw_shots, extract_ro_assignment_matrix, plot_ro_assignment_matrix,
                                                               get_ro_corrected_multi_probs)
from qi_utilities.utility_functions.data_handling import StoreProjectRecord, RetrieveProjectRecord
from qi_utilities.device_simulation.simulators import NoisySimulator # use the simulator in case all backends are unavailable!


provider = QIProvider()

print(provider.backends())

backend_name = "Tuna-9"
backend = provider.get_backend(name=backend_name)

print(backend)
#backend.coupling_map.draw()