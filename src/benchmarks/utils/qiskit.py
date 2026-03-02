import qiskit, torch
from qiskit_aer import Aer
import numpy as np
np.seterr(all='ignore')
from src.benchmarks.utils import safely_run_python

def setup_quantum_circuit(code_str:str, n_qubits:int, transpile:bool=False, max_seconds:float=10.0):
    func_code = f"""
def quantum_circuit():
    qc = QuantumCircuit({n_qubits})
{safely_run_python.indent_multiline_string(code_str, desired_indent=4)}
    return qc
"""    # to rectify indentation (breaks for loops etc. though) could use: "\n".join("    " + line.lstrip() for line in code_str.splitlines()) 
    quantum_circuit = safely_run_python.compile_restricted_function(
                    func_code,
                    "quantum_circuit",
                    {
                        "QuantumCircuit": qiskit.QuantumCircuit,
                        "Parameter": qiskit.circuit.Parameter,
                        "np": np,
                        "numpy": np,
                        "pi": np.pi,
                    }
                )
    qc = safely_run_python.execute_restricted_function(quantum_circuit, {}, max_seconds=max_seconds)
    if transpile:
        qc = qiskit.transpile(qc, basis_gates=['rz', 'rx', 'cx'], optimization_level=0)
    return qc

state_sim_backend = Aer.get_backend("aer_simulator_statevector")
state_sim_backend.set_options(device='CPU')
def quantum_circuit_state_vector(qc):
    qc = qc.copy()
    qc.save_statevector()
    sim_result = state_sim_backend.run(qc).result()
    return sim_result.get_statevector()

unitary_sim_backend = Aer.get_backend("aer_simulator_unitary")
unitary_sim_backend.set_options(device='CPU')
def quantum_circuit_unitary(qc):
    qc = qc.copy()
    sim_result = unitary_sim_backend.run(qc).result()
    return sim_result.get_unitary()

def exhaustive_pauli_observables(num_qubits:int, order:int):
    """
        Creates all pauli observables up to order `order` (excluding the all identity observable), i.e., up to `order` non-identity entries across all qubits.
    """
    assert order==2, "only order 2 implemented at the moment"
    obs = []
    # one-qubit terms
    for i in range(num_qubits):
        for pauli in ["X", "Y", "Z"]:
            z = ["I"] * num_qubits
            z[i] = pauli
            obs.append("".join(z))
    # two-qubit correlations
    for i in range(num_qubits):
        for pauli_i in ["X", "Y", "Z"]:
            for j in range(i+1, num_qubits):
                for pauli_j in ["X", "Y", "Z"]:
                    zz = ["I"] * num_qubits
                    zz[i], zz[j] = pauli_i, pauli_j
                    obs.append("".join(zz))
    return obs

def remove_global_phase(psi: torch.Tensor) -> torch.Tensor:
    """
        Removes global phase from torch tensor by multiplying with a complex number on the unit circle such that the first non-zero entry in `psi` becomes positive real (for normalized vectors this amounts to 1)
    """
    # Ensure psi is complex-valued
    if not torch.is_complex(psi):
        raise ValueError("Input must be a complex tensor.")
    # Find index of the first nonzero entry
    nonzero_mask = psi.abs() > 1e-12  # small tolerance
    if not torch.any(nonzero_mask):
        raise ValueError("Input state vector is all zeros.")
    idx = torch.where(nonzero_mask)[0][0]
    # Compute the global phase angle of the first nonzero element
    phase = torch.angle(psi[idx])
    # Remove global phase
    return psi * torch.exp(-1j * phase)

COMMON_GATES = [
    "x", "y", "z", "h", "s", "sdg", "t", "tdg",                                 # Clifford + T
    "rx", "ry", "rz",                                                           # rotation gates
    "u1", "u2", "u3",                                                           # universal single-qubit gates
    "cx", "cy", "cz", "swap", "crx", "cry", "crz", "rxx", "ryy", "rzz", "rzx",  # 2-qubit gates
    "cswap", "ccx",                                                             # 3-qubit gates
    "id", "sx", "sxdg",                                                         # misc
    ]