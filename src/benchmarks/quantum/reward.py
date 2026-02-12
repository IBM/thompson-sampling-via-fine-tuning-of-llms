from typing import List, Tuple
from functools import partial
import time, torch, re, scipy
import numpy as np
from transformers import AutoTokenizer
import src.benchmarks.utils.qiskit as qiskit_utils
import src.benchmarks.utils.safely_run_python as safely_run_python
from src import language_model
import src.benchmarks.quantum.features as q_features
import src.benchmarks.quantum.extract_hamiltonian as hamiltonians
from qiskit.quantum_info import Statevector, state_fidelity, SparsePauliOp
from qiskit import transpile

def quantum_state_preparation(tokenizer: AutoTokenizer, **kwargs):
    """
    Returns a state fidelity reward function that, given a quantum circuit, yields the fidelity to a pre-defined target state.

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the generation tokens (quantum circuit).

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument, encoding a quantum circuit, and returns the fidelity of the prepared state to a predefined but unknown state.
    """
    phi_AB = np.array([0, 1, 0, 0], dtype=complex) # |01>
    phi_CD = np.array([0, 1/np.sqrt(2), 1j/np.sqrt(2), 0], dtype=complex) # |01> + i|10>
    phi_EF = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex) # |01> + |10>
    phi_GH = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex) # |00> + |11>
    psi_4qubit_AD, psi_4qubit_EH = np.kron(phi_AB, phi_CD), np.kron(phi_EF, phi_GH)  # length 16 vectors
    psi_8qubit = np.kron(psi_4qubit_AD, psi_4qubit_EH)  # length 256 vector
    # Convert to Qiskit Statevector
    target_state = Statevector(psi_8qubit)

    def _qiskit(X: torch.Tensor):
        rewards = []
        for code_str in tokenizer.batch_decode(X, skip_special_tokens=True):
            try:
                code_str = safely_run_python.extract_code(code_str)
                qc = qiskit_utils.setup_quantum_circuit(code_str, n_qubits=8)
                test_state = qiskit_utils.quantum_circuit_state_vector(qc)
                #test_state = Statevector.from_instruction(qc)
                rewards.append(state_fidelity(target_state, test_state)) 
            except BaseException as e:
                print("Execution failed:", e)
                rewards.append(float('nan'))
        return torch.tensor(rewards, device="cpu", dtype=torch.double), [None] * len(rewards)
        
    return _qiskit

def quantum_hamiltonian(tokenizer: AutoTokenizer, hamiltonian_terms:List[Tuple[float, str]], **kwargs):
    """
    Returns a quantum hamiltonian reward function that, given a quantum circuit, yields the negative energy of the prepared state according to a pre-defined hamiltonian.

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the generation tokens (quantum circuit).
        hamiltonian_terms (List[Tuple[float, str]]): The hamiltonian expressed as a weighted sum of pauli operators

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument, encoding a quantum circuit, and returns the negative energy of the prepared state according to a pre-defined hamiltonian.
    """
    hamiltonian_coefficients = torch.tensor([coef for _, coef in hamiltonian_terms], device="cpu", dtype=torch.double)
    pauli_expectations = partial(q_features.quantum_observables, observables=[term for term, _ in hamiltonian_terms])
    # invalid codes gets assigned the ionization energy of 0 Hartree
    def _hamiltonian(X_tokens, **kwargs):
        state_observables = pauli_expectations(X_tokens, tokenizer).to("cpu").double() # last entry is 1 iff the code does not run
        rewards = - torch.matmul(state_observables[:, :len(hamiltonian_terms)], hamiltonian_coefficients)
        rewards[state_observables[:, -1] != 0.0] = float('nan')
        return rewards, [None] * rewards.numel()
    return _hamiltonian

h2_hamiltonian = partial(quantum_hamiltonian, hamiltonian_terms=hamiltonians.h2_pauli_terms)
ising = partial(quantum_hamiltonian, hamiltonian_terms=hamiltonians.ising_pauli_terms)
h2o = partial(quantum_hamiltonian, hamiltonian_terms=hamiltonians.h2o_pauli_terms)
artificial_hamiltonian = partial(quantum_hamiltonian, hamiltonian_terms=hamiltonians.artificial_pauli_terms)

def quantum_unitary_fidelity(tokenizer: AutoTokenizer, target_unitary:torch.Tensor, **kwargs):
    """
    Returns a circuit fidelity reward function that, given a quantum circuit, yields |Trace(target_unitary^H circuit_unitary)|/2^n_qubits

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the generation tokens (quantum circuit).
        target_unitary (torch.Tensor): The target unitary expressed as a complex tensor of shape (2^n_qubits, 2^n_qubits)

    flattened_unitary = unitary.flatten()
        return torch.cat([flattened_unitary.real, flattened_unitary.imag]) 

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument, encoding a quantum circuit, and returns |Trace(target_unitary^H circuit_unitary)|/2^n_qubits
    """
    assert target_unitary.ndim == 2 and target_unitary.shape[0] == target_unitary.shape[1]
    state_size = target_unitary.shape[0]
    def _quantum_unitary_fidelity(X_tokens, **kwargs):
        vectorized_unitaries = q_features.quantum_circuit_unitary(X_tokens, tokenizer)
        real_part, imag_part = vectorized_unitaries[:, :state_size], vectorized_unitaries[:, state_size:2*state_size]
        flattened = torch.complex(real_part, imag_part)
        batched_unitaries = torch.unflatten(flattened, dim=1, shape=(state_size, state_size))
        return torch.abs(torch.sum(target_unitary.conj().unsqueeze(0) * batched_unitaries, dim=(1, 2))) / state_size
    return _quantum_unitary_fidelity

def quantum_circuit_gate_expansion(tokenizer: AutoTokenizer, **kwargs):
    """
    Returns a gate efficiency reward function that, given a quantum circuit, yields the ratio of #gates_before_transpilation / #gates_after_transpilation

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the generation tokens (quantum circuit).

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument, encoding a quantum circuit, and returns the ratio of #gates_before_transpilation / #_gates_after_transpilation
    """

    def _gate_expansion(X: torch.Tensor):
        rewards = []
        for code_str in tokenizer.batch_decode(X, skip_special_tokens=True):
            try:
                code_str = safely_run_python.extract_code(code_str)
                m = re.search(r"qc = QuantumCircuit\(4\)(.*)", code_str, re.DOTALL)
                if m:
                    code_str = m.group(1) 
                qc = qiskit_utils.setup_quantum_circuit(code_str, n_qubits=4, transpile=False)
                n_gates_before_transpilation = qc.size()
                qc = transpile(qc, basis_gates=['rx', 'ry', 'cz'], optimization_level=0)
                n_gates_after_transpilation = qc.size()
                rewards.append(n_gates_before_transpilation / n_gates_after_transpilation) 
            except BaseException as e:
                print("Execution failed:", e)
                rewards.append(float('nan'))
        return torch.tensor(rewards, device="cpu", dtype=torch.double), [None] * len(rewards)
        
    return _gate_expansion

def variational_quantum_eigensolver(tokenizer: AutoTokenizer, **kwargs):
    """
    """
    paulis, coefs = zip(*hamiltonians.ising_pauli_terms)
    hamiltonian = SparsePauliOp(paulis, coefs)
    def _vqe(X: torch.Tensor):
        rewards = []
        for code_str in tokenizer.batch_decode(X, skip_special_tokens=True):
            try:
                code_str = safely_run_python.extract_code(code_str)
                m = re.search(r"qc = QuantumCircuit\(8\)(.*)", code_str, re.DOTALL)
                if m:
                    code_str = m.group(1) 
                ansatz = qiskit_utils.setup_quantum_circuit(code_str, n_qubits=8, transpile=False)
                parameters = list(ansatz.parameters)
                # inner loop to optimize suggested ansatz for hamiltonian at hand
                def energy(param_values: np.array):
                    param_dict = dict(zip(parameters, param_values)) # initialize to zero
                    bound_qc = ansatz.assign_parameters(param_dict)
                    state_vector = qiskit_utils.quantum_circuit_state_vector(bound_qc)
                    return state_vector.expectation_value(hamiltonian).real

                opt_result = scipy.optimize.minimize(energy, np.zeros(len(parameters)))
                rewards.append(-opt_result.fun)

            except BaseException as e:
                print("Execution failed:", e)
                rewards.append(float('nan'))
        return torch.tensor(rewards, device="cpu", dtype=torch.double), [None] * len(rewards)
    return _vqe

if __name__ == "__main__":
    tokenizer = language_model.get_tokenizer("Qwen/Qwen3-0.6B")
    circuit = """
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
""" 
    tokenized_prompts = tokenizer([circuit], padding=True, return_tensors="pt")['input_ids']
    start_time = time.perf_counter()
    reward, _, _ = quantum_state_preparation(tokenizer)(tokenized_prompts)
    print("TIME TAKEN FOR EVALUATION", time.perf_counter() - start_time) 
    print(f"the qiskit reward is {reward}")

   