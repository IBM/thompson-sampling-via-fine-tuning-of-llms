from typing import List, Callable
from functools import partial
import torch
from transformers import AutoTokenizer
import src.benchmarks.utils.qiskit as qiskit_utils
import src.benchmarks.utils.safely_run_python as safely_run_python
from src.utils import misc
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Statevector, SparsePauliOp
import src.benchmarks.quantum.extract_hamiltonian as extrac_hamiltonian

def construct_quantum_feature_map(X_tokens:torch.Tensor, tokenizer:AutoTokenizer, n_qubits:int, n_features:int, feature_extraction:Callable):
    features = []
    for code_str in tokenizer.batch_decode(X_tokens, skip_special_tokens=True):
        try:
            code_str = safely_run_python.extract_code(code_str)
            qc = qiskit_utils.setup_quantum_circuit(code_str, n_qubits=n_qubits, transpile=False, max_seconds=10)
            feature_map = feature_extraction(qc)
            feature_map = torch.cat([feature_map, torch.zeros((1))]).to(X_tokens.device) # add a dummy dimension for invalid qiskit code
            assert torch.isfinite(feature_map).all()
            assert qc.num_qubits == n_qubits, "the provided code changed the number of qubits which is disallowed"
            assert feature_map.numel() == n_features + 1, "the generated feature map must be of the same shape as the parameter `n_features` to ensure embedding dimensionality consistent with invalid code"
        except BaseException as e: # note: also catches keyboard interrupt
            print("Invalid Circuit Feature:", e)
            print("when running the code", code_str)
            feature_map = torch.zeros((n_features+1), device=X_tokens.device)
            feature_map[-1] = 1.0 # indicate invalid code with dummy dimension 
        features.append(feature_map)
    return torch.stack(features) # (batch_size, n_features) where n_features= # observables + 1  


def quantum_observables(X_tokens:torch.Tensor, tokenizer:AutoTokenizer, observables:List[str], **kwargs):
    """
        Computes the expected value of the provided quantum observables on the state vector resulting from running a quantum simulation
    """
    def extraction(qc):
        state = qiskit_utils.quantum_circuit_state_vector(qc)
        #state = Statevector.from_instruction(qc)
        return torch.tensor([state.expectation_value(SparsePauliOp([obs])).real for obs in observables]) 
    return construct_quantum_feature_map(X_tokens, tokenizer, n_qubits=len(observables[0]), n_features=len(observables), feature_extraction=extraction)

h2_hamiltonian_observables = partial(quantum_observables, observables=[term for term, _ in extrac_hamiltonian.h2_pauli_terms])
ising_hamiltonian_observables = partial(quantum_observables, observables=[term for term, _ in extrac_hamiltonian.ising_pauli_terms])
h2o_hamiltonian_observables = partial(quantum_observables, observables=[term for term, _ in extrac_hamiltonian.h2o_pauli_terms])
artificial_hamiltonian_observables = partial(quantum_observables, observables=[term for term, _ in extrac_hamiltonian.artificial_pauli_terms])

def quantum_state(X_tokens:torch.Tensor, tokenizer:AutoTokenizer, num_qubits:int, **kwargs):
    """
        Computes the quantum state in stacked real vector format (first real entries, then imaginary entries) upon application of the quantum circuit on the zero-initialized qubits.
    """
    def extraction(qc):
        state = qiskit_utils.quantum_circuit_state_vector(qc)
        #state = Statevector.from_instruction(qc)
        state_tensor = qiskit_utils.remove_global_phase(torch.tensor(state.data, device=X_tokens.device, dtype=torch.cfloat))
        return torch.cat([state_tensor.real, state_tensor.imag])
        #density_matrix = torch.outer(real_tensor, real_tensor) # torch.flatten(density_matrix)
    return construct_quantum_feature_map(X_tokens, tokenizer, n_qubits=num_qubits, n_features=2*(2**num_qubits), feature_extraction=extraction)

def quantum_circuit_unitary(X_tokens:torch.Tensor, tokenizer:AutoTokenizer, num_qubits:int, **kwargs):
    """
        Computes the unitary matrix that corresponds to the quantum circuit, flattens it, and concatenates the real and imaginary parts. All in batched fashion.
    """
    def extraction(qc):
        unitary = torch.tensor(qiskit_utils.quantum_circuit_unitary(qc))
        flattened_unitary = unitary.flatten()
        return torch.cat([flattened_unitary.real, flattened_unitary.imag]) 
    return construct_quantum_feature_map(X_tokens, tokenizer, n_qubits=num_qubits, n_features=2*(2**(2*num_qubits)), feature_extraction=extraction)
    

def quantum_circuit_features(X_tokens:torch.Tensor, tokenizer:AutoTokenizer, **kwargs):
    """
    Extracts an embedding roughly describing the quantum circuit.

    Args:
        X_tokens (torch.Tensor): The batched input tokens describing the quantum circuit.
        tokenizer (AutoTokenizer): The tokenizer that converts between text and tokens.

    Returns:
        out (torch.Tensor): A tensor containing the batched circuit features.
    """
    def extraction(qc):
        # Get basic structural properties
        num_qubits = qc.num_qubits
        depth = qc.depth()
        # get a hash count for uniqueness
        hash = misc.norm_hash(qc.qasm())
        # Get gate counts
        ops = qc.count_ops()
        gate_count = [ops.get(gate, 0.0) for gate in qiskit_utils.COMMON_GATES]
        return torch.tensor([num_qubits, depth, hash] + gate_count, device=X_tokens.device)
    return construct_quantum_feature_map(X_tokens, tokenizer, n_qubits=8, n_features=3+len(qiskit_utils.COMMON_GATES)+1, feature_extraction=extraction)