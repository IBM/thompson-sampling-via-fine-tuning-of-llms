import math, time, torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import language_model, kernel_features 

def sample_and_extract_generator_features(total_prompt: str,
                                          tokenizer:AutoTokenizer, 
                                          generator:AutoModelForCausalLM,
                                          temperature:float, 
                                          max_new_tokens:int, 
                                          num_samples:int, 
                                          batch_size:int,
                                          hidden_dim:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples from a generator and returns the sampled tokens, the log generation probability, and the (across text length) aggregated feature embeddings of
    the token-embedding, the penultimate hidden layer, and the hidden layer according to both the `sequence_mean` and the `sequence_latest` aggregation method.

    Args:
        reward_function (Callable): Black-box function that takes a generated output (batched sequence of tokens in torch.Tensor format) and returns its associated reward, validity, and correctness.
        total_prompt (str): The prompt that is used to induce the sampling process. For instruction-tuned models this includes the system prompt.
        tokenizer (AutoTokenizer): Tokenizer that maps from text to tokens and vice-versa. Should match the `generator`.
        generator (AutoModelForCausalLM): Pre-trained language model used as the policy from which we sample.
        temperature (float): Temperature parameter for stochastic sampling from the generator.
        max_new_tokens (int): Maximum length of LLM generation.
        num_samples (int): Number of samples to generate.
        batch_size (int): Batch size according to which the samples are drawn from the generator.
        hidden_dim (int): The hidden dimension of the generator.

    Returns:
        out (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The generated tokens, their log generation probability, the mean-token-embedding, the mean-penultimate-hidden states, the mean-last-hidden states, the latest-penultimate-hidden states, and the latest-last-hidden states.
    """ 
    X_tokens = torch.full((num_samples, max_new_tokens), tokenizer.eos_token_id, device=generator.device)
    X_log_prob = torch.full((num_samples,), float('nan'), device=generator.device, dtype=torch.float32)
    X_features_mte = torch.full((num_samples, hidden_dim), float('nan'), device=generator.device)
    X_features_mph = torch.full((num_samples, hidden_dim), float('nan'), device=generator.device)
    X_features_mlh = torch.full((num_samples, hidden_dim), float('nan'), device=generator.device)
    X_features_lph = torch.full((num_samples, hidden_dim), float('nan'), device=generator.device)
    X_features_llh = torch.full((num_samples, hidden_dim), float('nan'), device=generator.device)
    # sample from generator in batches
    prompt_ids = tokenizer(total_prompt, return_tensors="pt")['input_ids']
    for step in range(0, batch_size * int(math.ceil(num_samples / batch_size)), batch_size):
        print(f"STEP {step}/{num_samples}")
        start_time = time.perf_counter()
        with torch.no_grad():
            new_X, new_X_log_prob, new_X_hidden_states = language_model.generate_tokens_with_log_probs_and_hidden_states(prompt_ids, 
                                                                                                                         generator, 
                                                                                                                         tokenizer, 
                                                                                                                         ..., 
                                                                                                                         temperature, 
                                                                                                                         min(batch_size, num_samples - step), 
                                                                                                                         max_new_tokens,
                                                                                                                         verbose=True)
        print(f"TIMING: {time.perf_counter() - start_time :.3f} seconds to sample batch of size {batch_size} from LLM")
        # extract all considered feature combinations from the hidden states
        new_X_features_mte = kernel_features.sequence_mean(new_X_hidden_states[0], new_X, tokenizer)
        new_X_features_mph = kernel_features.sequence_mean(new_X_hidden_states[-2], new_X, tokenizer)
        new_X_features_mlh = kernel_features.sequence_mean(new_X_hidden_states[-1], new_X, tokenizer)
        new_X_features_lph = kernel_features.sequence_latest(new_X_hidden_states[-2], new_X, tokenizer)
        new_X_features_llh = kernel_features.sequence_latest(new_X_hidden_states[-1], new_X, tokenizer)

        assert torch.isfinite(new_X_features_mte).all(), f"tensor of shape {new_X_features_mte.shape} and data type {new_X_features_mte.dtype} has only {torch.isfinite(new_X_features_mte).sum().item()} finite entries"
        assert torch.isfinite(new_X_features_mph).all(), f"tensor of shape {new_X_features_mph.shape} and data type {new_X_features_mph.dtype} has only {torch.isfinite(new_X_features_mph).sum().item()} finite entries"
        assert torch.isfinite(new_X_features_mlh).all(), f"tensor of shape {new_X_features_mlh.shape} and data type {new_X_features_mlh.dtype} has only {torch.isfinite(new_X_features_mlh).sum().item()} finite entries"
        assert torch.isfinite(new_X_features_lph).all(), f"tensor of shape {new_X_features_lph.shape} and data type {new_X_features_lph.dtype} has only {torch.isfinite(new_X_features_lph).sum().item()} finite entries"
        assert torch.isfinite(new_X_features_llh).all(), f"tensor of shape {new_X_features_llh.shape} and data type {new_X_features_llh.dtype} has only {torch.isfinite(new_X_features_llh).sum().item()} finite entries"
        
        # store results
        X_tokens[step:step+batch_size, :new_X.shape[1]] = new_X # everything after token_length will be initialized to <eos>
        X_log_prob[step:step+batch_size] = new_X_log_prob
        X_features_mte[step:step+batch_size, :] = new_X_features_mte
        X_features_mph[step:step+batch_size, :] = new_X_features_mph
        X_features_mlh[step:step+batch_size, :] = new_X_features_mlh
        X_features_lph[step:step+batch_size, :] = new_X_features_lph
        X_features_llh[step:step+batch_size, :] = new_X_features_llh

    return X_tokens, X_log_prob, X_features_mte, X_features_mph, X_features_mlh, X_features_lph, X_features_llh