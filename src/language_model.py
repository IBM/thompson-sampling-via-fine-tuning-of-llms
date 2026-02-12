from typing import Tuple
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model(model_specification:str) -> AutoModelForCausalLM:
    """
    Loads a pre-trained transformer model based on the given specification string.

    Args:
        model_specification (str): The specifier string of the model hosted on huggingface.

    Returns:
        out (AutoModelForCausalLM): An instance of the loaded transformer model.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_specification, cache_dir="./model_weights/", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True) # assumes that it is called from directory tosfit
    except Exception as e: # handles the case where flash_attention is not implemented for the model
        print(e)
        model = AutoModelForCausalLM.from_pretrained(model_specification, cache_dir="./model_weights/", torch_dtype=torch.bfloat16, trust_remote_code=True) # assumes that it is called from directory tosfit
    return model

def get_tokenizer(tokenizer_specification:str) -> AutoTokenizer:
    """
    Loads a tokenizer based on the given specification string.

    Args:
        tokenizer_specification (str): The specifier string of the tokenizer hosted on huggingface.

    Returns:
        out (AutoTokenizer): An instance of the loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_specification, cache_dir="./model_weights/", use_fast=False, trust_remote_code=True, padding_side="left") # assumes that it is called from directory tosfit

def unfreeze_last_n_layers(model:AutoModelForCausalLM, n:int):
    """
    A function to unfreeze the last n layers of a AutoModelForCausalLM model.

    Args:
        model (AutoModelForCausalLM): The AutoModelForCausalLM model.
        n (int): The number of last layers to unfreeze.
    """
    # 1. Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Identify the transformer blocks
    transformer_blocks = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_blocks = model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_blocks = model.model.layers
    elif hasattr(model, 'layers'):
        transformer_blocks = model.layers
    else:
        # Add more checks for other model architectures if needed
        raise ValueError("Could not identify the transformer blocks in the model.")

    # 3. Unfreeze the last n layers
    if n > 0 and transformer_blocks:
        for layer in transformer_blocks[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

def generate_tokens_exluding_backprop(prompt_ids:torch.Tensor, 
                                      model:AutoModelForCausalLM, 
                                      tokenizer:AutoTokenizer, 
                                      temperature:float=1.0, 
                                      batch_size:int=4, 
                                      max_new_tokens:int=1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates token sequences and their log-probabilities using the model without backpropagation. Due to rounding errors, 
    this log-probability may differ from the non-autoregressively computed ones at `log_probs_and_hidden_states` or `low_memory_log_probs_and_hidden_states`.

    Args:
        prompt_ids (torch.Tensor): The initial input tokens as a tensor.
        model (AutoModelForCausalLM): The transformer model used for generation.
        tokenizer (AutoTokenizer): The tokenizer to convert tokens to strings and vice versa.
        temperature (float, optional): Temperature parameter for sampling. Default: 1.0
        batch_size (int, optional): Number of sequences to generate. Default: 4
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Default: 1024

    Returns:
        out (Tuple[torch.Tensor, torch.Tensor]): A tensor of shape (batch_size, max_sequence_length) containing the generated token sequences and a tensor of shape (batch_size,) containing the associated log probabilities.
    """
    prompt_ids = prompt_ids.to(model.device)
    with torch.no_grad(): # does not track the computational graph, which is essential to keep memory bounded during autoregressive generation using key/value caches.
        log_generation_probabilities = torch.zeros(batch_size, device=model.device, dtype=torch.double)
        tokens_list = [prompt_ids.expand(batch_size, -1)] # excludes the non-representable continuous tokens
        past_key_values = None # cache
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=model.device) # tracks early exit
        for _ in range(max_new_tokens-1):
            outputs = model(input_ids=tokens_list[-1], past_key_values=past_key_values, output_hidden_states=False, use_cache=True, return_dict=True)
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size) for the last token in each sequence (only really relevant when feeding in prompt)
            if not logits.isfinite().all():
                print("WARNING! there are invalid logits in autoregressive generation. This is likely caused by model divergence. Try to reduce the learning rate and increase the batch size.")
                continue
            past_key_values = outputs.past_key_values
            # token sampling
            epsilon = torch.finfo(logits.dtype).eps
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + epsilon) + epsilon)
            gumbel_logits = (logits / temperature) + gumbel_noise
            next_tokens = gumbel_logits.argmax(dim=-1)  # (batch_size,)
            #assert gumbel_logits[torch.arange(batch_size), next_tokens[torch.arange(batch_size)]].isfinite().all(), f"there is an invalid logit in batched 'next token logits' {gumbel_logits[torch.arange(batch_size), next_tokens[torch.arange(batch_size)]]}"
            log_generation_probabilities += unfinished_sequences * torch.gather(input=torch.nn.functional.log_softmax(logits/temperature, dim=-1), dim=-1, index=next_tokens.unsqueeze(-1))[:, 0]
            next_tokens[~unfinished_sequences] = tokenizer.eos_token_id
            unfinished_sequences &= next_tokens != tokenizer.eos_token_id
            # extend sequence and check for early exit
            tokens_list += [next_tokens.unsqueeze(1)]
            if not unfinished_sequences.any():
                break
        if unfinished_sequences.any():
            tokens_list += [torch.full((batch_size, 1), tokenizer.eos_token_id, device=model.device)] # always ends with an <eos> token
        all_tokens = torch.cat(tokens_list, dim=1) # at this stage all generated sequences have an <eos> at the end, possibly forced due to the max_new_tokens restriction
        print("MAX TOKENS IN CONVERSATION: ", all_tokens.shape[1])
        generated_tokens = all_tokens[:, prompt_ids.numel():]
        return generated_tokens, log_generation_probabilities

def compute_log_probability(generated_token_ids:torch.tensor, 
                            logits:torch.tensor, 
                            temperature:float, 
                            tokenizer:AutoTokenizer) -> torch.Tensor:
    """
    Computes the log probability of a sequence of tokens given their logits.

    Args:
        generated_token_ids (torch.Tensor): The token IDs produced by generation. Shape (batch_size, max_generated_sequence_length).
        logits (torch.Tensor): The unnormalized log probabilities for each token. Shape (batch_size, max_generated_sequence_length, vocabulary).
        temperature (float): Temperature parameter to adjust sampling sharpness. Default: 1.0
        tokenizer (AutoTokenizer): Tokenizer used for converting tokens and handling padding.

    Returns:
        out (torch.Tensor): A tensor containing the computed log probabilities. Shape (batch_size,).
    """
    assert generated_token_ids.shape[1] == logits.shape[1], f"shape mismatch where generated_token_ids.shape = {generated_token_ids.shape} and logits.shape = {logits.shape}"
    generated_token_ids = generated_token_ids.to(logits.device)
    scores = torch.nn.functional.log_softmax(logits/temperature, dim=-1) # log transition probabilities
    selected_scores = torch.gather(input=scores, dim=-1, index=generated_token_ids.unsqueeze(2))[:, :, 0]
    # sum over sequence length until <eos_token>, does not sum over batch length. Also works if <eos_token> = <pad_token> 
    eos_mask = torch.cumsum(generated_token_ids == tokenizer.eos_token_id, dim=-1) > 1 # removes first occurence of <eos_token> from the mask
    selected_scores[eos_mask] = 0
    return selected_scores.double().sum(axis=-1) # reduction operation needs high precision to avoid rounding errors

def log_probs_and_hidden_states(prompt_ids:torch.Tensor,
                                generated_ids:torch.Tensor, 
                                model:AutoModelForCausalLM, 
                                tokenizer:AutoTokenizer, 
                                hidden_layer:int,
                                temperature:float=1.0,
                                max_new_tokens:int=1024,
                                print_memory:bool=False) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
    """
    Computes log probabilities and hidden states for the generated token sequences. Due to rounding errors, this log-probability
    may differ from the autoregressively computed ones at `generate_tokens_exluding_backprop`

    Args:
        prompt_ids (torch.Tensor): Initial input tokens.
        generated_ids (torch.Tensor): Tokens (presumably) generated by the model.
        model (AutoModelForCausalLM): The transformer model (presumably) used for generation.
        tokenizer (AutoTokenizer): Tokenizer to convert between tokens and strings.
        hidden_layer (int): The hidden layer to extract and return. If set to None, memory is saved and None is returned. If set to ... (Ellipsis), all layers are returned as a tuple.
        temperature (float, optional): Temperature parameter (presumably) used for sampling. Default: 1.0
        max_new_tokens (int, optional): The maximum number of new tokens that could have been generated. Used to correct the probability computation for the forced <eos> token at the maximum length. Default: 1024
        print_memory (bool, optional): Whether to print memory usage. Default: False

    Returns:
        out (tuple[torch.Tensor, tuple[torch.Tensor]]): A tuple containing log probabilities (shape (batch_size,)) and corresponding hidden states (shape (batch_size, max_generated_sequence_length, hidden_size), or tuple of those shapes if hidden_layer = -1).
    """
    assert generated_ids.ndim == 2 and generated_ids.shape[1] <= max_new_tokens, f"generated_ids.shape = {generated_ids.shape} and max_new_tokens = {max_new_tokens}"
    prompt_ids, generated_ids = prompt_ids.to(model.device), generated_ids.to(model.device)
    prefix_length = prompt_ids.numel()
    total_tokens = torch.cat((prompt_ids.expand(generated_ids.shape[0], -1), generated_ids), dim=1) # concatenates along sequence
    # pass the tokens through the model to get the log probabilities and hidden states for each token in the sequence (including prompt) 
    model.gradient_checkpointing_enable() # reduces memory footprint but slows down model
    outputs = model(input_ids=total_tokens, output_hidden_states=hidden_layer is not None, past_key_values=None, use_cache=False, return_dict=True)
    model.gradient_checkpointing_disable()
    # extract hidden states and logits that would have generated tokens - note that the logits of interest are offset to the left w.r.t. the tokens they would generate
    if hidden_layer is None:
        hidden_states = None
    elif hidden_layer is ...:
        hidden_states = tuple(hs[:, prefix_length:, :] for hs in outputs.hidden_states)
    else:
        hidden_states = outputs.hidden_states[hidden_layer][:, prefix_length:, :]
    logits = outputs.logits[:, prefix_length-1:-1, :]
    if generated_ids.shape[1] == max_new_tokens:
        generated_ids, logits = generated_ids[:, :-1], logits[:, :-1, :] # remove forced <eos> token probability from log-probability computation to ensure correct computation
    log_probs = compute_log_probability(generated_ids, logits, temperature, tokenizer)
    if print_memory:
        memory_usages = (sum([hs.numel() * hs.element_size() for hs in outputs.hidden_states]) if hidden_layer is not None else 0) + outputs.logits.numel() * outputs.logits.element_size()
        print(f"Memory usage of hidden states and logits across a batch of {outputs.logits.shape[0]} sequences of length {outputs.logits.shape[1]}: {memory_usages / 1.0E9} Gigabytes.")
    return log_probs, hidden_states # hidden states excluding prompt. May include a bunch of <eos> tokens

def low_memory_log_probs_and_hidden_states(mini_batch_size: int,
                                           prompt_ids:torch.Tensor,
                                           generated_ids:torch.Tensor, 
                                           model:AutoModelForCausalLM, 
                                           tokenizer:AutoTokenizer, 
                                           hidden_layer:int,
                                           temperature:float=1.0,
                                           max_new_tokens:int=1024,
                                           print_memory:bool=False) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
    """
    Computes log probabilities and hidden states for the generated token sequences in a memory-conscious manner. Due to rounding errors, this log-probability
    may differ from the autoregressively computed ones at `generate_tokens_exluding_backprop`. Note that the memory savings may not be particularly pronounced 
    unless autograd is turned off.

    Args:
        mini_batch_size (int): Mini batch size for finetuning, where the batch is split into mini-batches of size `mini_batch_size` which are consecutively backpropagated through and used to update the parameters. Note that all mini-batches share the same (old) policy. Small mini-batch size helps with memory issues.
        prompt_ids (torch.Tensor): Initial input tokens.
        generated_ids (torch.Tensor): Tokens (presumably) generated by the model.
        model (AutoModelForCausalLM): The transformer model (presumably) used for generation.
        tokenizer (AutoTokenizer): Tokenizer to convert between tokens and strings.
        hidden_layer (int): The hidden layer to extract and return. If set to None, memory is saved and None is returned. If set to ... (Ellipsis), all layers are returned as a tuple.
        temperature (float, optional): Temperature parameter (presumably) used for sampling. Default: 1.0
        max_new_tokens (int, optional): The maximum number of new tokens that could have been generated. Used to correct the probability computation for the forced <eos> token at the maximum length. Default: 1024
        print_memory (bool, optional): Whether to print memory usage. Default: False

    Returns:
        out (tuple[torch.Tensor, tuple[torch.Tensor]]): A tuple containing log probabilities (shape (batch_size,)) and corresponding hidden states (shape (batch_size, max_generated_sequence_length, hidden_size), or tuple of those shapes if hidden_layer = -1).
    """
    prompt_ids, generated_ids = prompt_ids.to(model.device), generated_ids.to(model.device)
    log_probs, hidden_layers = [], []
    for mini_batch_idx in range(math.ceil(generated_ids.shape[0]/mini_batch_size)):
        mini_batch_generated_tokens = generated_ids[mini_batch_idx*mini_batch_size:(mini_batch_idx+1)*mini_batch_size, :]
        mini_batch_log_probs, mini_batch_hidden_layer = log_probs_and_hidden_states(prompt_ids, mini_batch_generated_tokens, model, tokenizer, hidden_layer, temperature, max_new_tokens, print_memory)
        log_probs.append(mini_batch_log_probs)
        hidden_layers.append(mini_batch_hidden_layer)
    log_probs = torch.cat(log_probs, dim=0)
    if hidden_layer is None:
        hidden_layers = None
    elif hidden_layer is ...:
        hidden_layers = tuple(torch.cat(tensors, dim=0) for tensors in zip(*hidden_layers))
    else:
        hidden_layers = torch.cat(hidden_layers, dim=0)
    return log_probs, hidden_layers

def generate_tokens_with_log_probs_and_hidden_states(prompt_ids:torch.Tensor, 
                                                     model:AutoModelForCausalLM, 
                                                     tokenizer:AutoTokenizer, 
                                                     hidden_layer: int,
                                                     temperature:float=1.0, 
                                                     batch_size:int=4, 
                                                     max_new_tokens:int=1024,
                                                     verbose:bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates token sequences and computes their log probabilitie.

    Args:
        prompt_ids (torch.Tensor): Initial input tokens.
        model (AutoModelForCausalLM): The transformer model used for generation.
        tokenizer (AutoTokenizer): Tokenizer to convert between tokens and strings.
        hidden_layer (int): The hidden layer to extract and return. If set to None, memory is saved and None is returned. If set to ... (Ellipsis), all layers are returned as a tuple.
        temperature (float, optional): Temperature parameter for token sampling during generation. Default: 1.0
        batch_size (int, optional): Number of sequences to generate. Default: 4
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Default: 1024
        verbose (bool, optional): If True, prints detailed information about the generated response.

    Returns:
        out (tuple[torch.Tensor, torch.Tensor]): A tuple containing the generated token sequences (shape (batch_size, max_generated_sequence_length)), log probabilities (shape (batch_size,)) and hidden states (shape (batch_size, max_generated_sequence_length, hidden_size), or tuple of those shapes if hidden_layer = -1).
    """
    prompt_ids = prompt_ids.to(model.device)
    generated_tokens, log_probs_ = generate_tokens_exluding_backprop(prompt_ids, model, tokenizer, temperature, batch_size, max_new_tokens)
    log_probs, hidden_layers = log_probs_and_hidden_states(prompt_ids, generated_tokens, model, tokenizer, hidden_layer, temperature, max_new_tokens)
    print(f"the difference between the autoregressively computed log probabilities and the non-autoregressively computed ones is {log_probs_-log_probs}. This seems to be a rounding error.")
    if verbose:
        first_response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0]
        print(f"Example of a generated response:\n{first_response}")
    return generated_tokens, log_probs, hidden_layers # hidden states are one dimension shorter because they exclude <eos> token.