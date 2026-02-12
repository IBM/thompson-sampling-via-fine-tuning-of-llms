from src import benchmarks

shortness_of_response = benchmarks.text.reward.shortness_of_response
shakespearity = benchmarks.text.reward.shakespearity
faq = benchmarks.text.reward.faq
bin_packing_excess_bins = benchmarks.code.reward.bin_packing_excess_bins
math_correctness = benchmarks.math.reward.math_correctness
aime2024_prompt_optimization = benchmarks.prompt.reward.aime2024_prompt_optimization
generate_and_judge = benchmarks.prompt.reward.generate_and_judge
protein_stability = benchmarks.biology.reward.protein_stability
quantum_state_preparation = benchmarks.quantum.reward.quantum_state_preparation
artificial_quantum_hamiltonian = benchmarks.quantum.reward.artificial_hamiltonian
quantum_circuit_gate_expansion = benchmarks.quantum.reward.quantum_circuit_gate_expansion
vqe = benchmarks.quantum.reward.variational_quantum_eigensolver

reward_for_invalid_generation = {
    shortness_of_response: 0.0,
    shakespearity: 0.0,
    faq: 0.0,
    bin_packing_excess_bins: 1.0,
    math_correctness: 0.0,
    aime2024_prompt_optimization: 0.0,
    generate_and_judge: 0.0,
    protein_stability: -449.4, # lowest possible thermal stability scores. Highest is 159.1
    quantum_state_preparation: -1.0,
    artificial_quantum_hamiltonian: - 7.0,
    quantum_circuit_gate_expansion: 0.0,
    vqe: 0.0, # free energy
}