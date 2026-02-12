import re, torch
from functools import reduce, partial
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from src import kernel_features
import src.benchmarks.quantum.features as q_features
import src.benchmarks.utils.qiskit as qiskit_utils

def feature_map(config):
    # Configure the embedding model and feature transformation for the kernel in the reward model
    n_features = config.get("embedding_dim", config['hidden_dim'])
    _feature_embedding_model = config['feature_embedding_model']
    match _feature_embedding_model:
        case "token_embedding":
            feature_embedding_model = 0 # first layer from reference_generator
        case "penultimate_hidden":
            feature_embedding_model = -2 # penultimate layer from reference_generator
        case "last_hidden":
            feature_embedding_model = -1 # last layer from reference_generator
        case "quantum_circuit_features":
            feature_embedding_model = q_features.quantum_circuit_features
        case "quantum_state":
            feature_embedding_model = partial(q_features.quantum_state, num_qubits=8)
        case "h2_observables":
            feature_embedding_model = q_features.h2_hamiltonian_observables
        case "ising_observables":
            feature_embedding_model = q_features.ising_hamiltonian_observables
        case "h2o_observables":
            feature_embedding_model = q_features.h2o_hamiltonian_observables
        case "artificial_observables":
            feature_embedding_model = q_features.artificial_hamiltonian_observables
        case "pauli_observables":
            feature_embedding_model = partial(q_features.quantum_observables, observables=qiskit_utils.exhaustive_pauli_observables(num_qubits=7, order=2)) 
        case "qwen3_embedding_0.6B":
            embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device_map": "auto"}, tokenizer_kwargs={"padding_side": "left"}) #"attn_implementation": "flash_attention_2",
            def handler(prompt_ids:torch.Tensor, X_tokens:torch.Tensor, tokenizer:AutoTokenizer):
                completions = tokenizer.batch_decode(X_tokens, skip_special_tokens=True)
                return embedding_model.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
            feature_embedding_model = handler
        case "qwen3_embedding_0.6B_256":
            embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device_map": "auto"}, tokenizer_kwargs={"padding_side": "left"}, truncate_dim=256) #"attn_implementation": "flash_attention_2",
            def handler(prompt_ids:torch.Tensor, X_tokens:torch.Tensor, tokenizer:AutoTokenizer):
                completions = tokenizer.batch_decode(X_tokens, skip_special_tokens=True)
                return embedding_model.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
            feature_embedding_model = handler 
        case "qwen3_embedding_0.6B_32":
            embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device_map": "auto"}, tokenizer_kwargs={"padding_side": "left"}, truncate_dim=32) #"attn_implementation": "flash_attention_2",
            def handler(prompt_ids:torch.Tensor, X_tokens:torch.Tensor, tokenizer:AutoTokenizer):
                completions = tokenizer.batch_decode(X_tokens, skip_special_tokens=True)
                return embedding_model.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
            feature_embedding_model = handler 
        case _:
            assert False, f"the feature_embedding_model {_feature_embedding_model} has no matching implementation"
    _embedding_aggregation = config['embedding_aggregation']
    match _embedding_aggregation:
        case "mean":
            embedding_aggregation = kernel_features.sequence_mean
        case "latest":
            embedding_aggregation = kernel_features.sequence_latest
        case "nop":
            embedding_aggregation = lambda features, X, tokenizer: features
        case _:
            assert False, f"the embedding_aggregation {_embedding_aggregation} has no matching implementation"
    kernel_feature_transformations_list = []
    for step_kernel_feature_transformation in config['kernel_feature_transformation'].split("-"):
        match step_kernel_feature_transformation:
            case "normalize":
                kernel_feature_transformations_list.append(kernel_features.normalize)
            case _ if re.fullmatch(r"rff\([\d.,\s]+\)", step_kernel_feature_transformation): # mathes e.g. rff(0.5, 10000)
                match = re.search(r"[\d.,\s]+", step_kernel_feature_transformation)
                length_scale, n_rffs = tuple(map(float, match.group().split(',')))
                n_rffs = int(n_rffs)
                kernel_feature_transformations_list.append(lambda fs: kernel_features.random_fourier_features(fs, rff_parameters=(length_scale, n_rffs)))
                n_features = n_rffs
            case "bias":
                kernel_feature_transformations_list.append(kernel_features.add_bias)
                n_features = n_features + 1
            case "nop":
                kernel_feature_transformations_list.append(lambda f:f)
                n_features = n_features
            case _:
                assert False, f"the kernel_feature_transformation {step_kernel_feature_transformation} has no matching implementation"
    # step_kernel_feature_transformation = normalize-rff(0.5, 10000) would be translated to kernel_feature_transformation(fs) = rff(normalize(fs), (0.5, 10000))
    kernel_feature_transformation = lambda fs: reduce(lambda acc, func: func(acc), kernel_feature_transformations_list, fs)
    return feature_embedding_model, embedding_aggregation, kernel_feature_transformation, n_features