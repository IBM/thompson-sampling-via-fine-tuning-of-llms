from typing import Callable, Tuple, List
from functools import partial
import torch, math, re
from transformers import AutoModel, AutoTokenizer

def _make_step_rewards(logits, token_masks):
        """
            Takes a batched sequence of logits (shape B,L,2) as well as a matching token mask (shape B,L) and
            extracts the probability of correctness of each step, returning a correctness tensor of shape B,L.
        """
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # (bs, seq_len, num_labels), invalid tokens are assigned probability 0
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels -> valid_tokens
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

process_reward_model_tokenizer = process_reward_model = None

def _math_correctness(X:torch.Tensor, prm:str, tokenizer:AutoTokenizer, max_new_tokens:int, system_prompt:str, prompt:str, answer:str, device:torch.device):
    global process_reward_model_tokenizer, process_reward_model
    if process_reward_model_tokenizer is None:
        process_reward_model_tokenizer = AutoTokenizer.from_pretrained(prm, trust_remote_code=True)
        if prm in ["Qwen/Qwen2.5-Math-PRM-72B"]: # too big for one GPU
            process_reward_model = AutoModel.from_pretrained(prm, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto').eval() 
        else:
            process_reward_model = AutoModel.from_pretrained(prm, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval() 

    batched_generation = tokenizer.batch_decode(X, skip_special_tokens=True)
    conversation_strings = []
    for generation in batched_generation:
        data = {
            "system": system_prompt,
            "query": prompt,
            "response": generation.split("\n\n")
        }
        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"}, # the <extra_0> token will be used to extract probability of step correctness
        ]
        conversation_strings.append(process_reward_model_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        ))
    assert conversation_strings is not None, "conversation_strings is None!"
    input_ids = process_reward_model_tokenizer(conversation_strings, padding=True, return_tensors="pt")['input_ids'].to(device)
    with torch.inference_mode():
        outputs = process_reward_model(input_ids=input_ids)
    step_sep_id = process_reward_model_tokenizer("<extra_0>")['input_ids'][0]
    token_masks = (input_ids == step_sep_id)
    batched_step_rewards = _make_step_rewards(outputs[0], token_masks) # outputs[0] extracts logits
    batched_total_rewards = torch.tensor([math.prod(step_rewards) for step_rewards in batched_step_rewards], dtype=torch.double, device="cpu")
    # check ground-truth correctness
    extracted_answers = []
    for response in batched_generation:
        match = re.search(r'\\boxed\{(.*?)\}', response)
        boxed_content = match.group(1) if match else None
        number_match = re.search(r'(\d+)', boxed_content) if boxed_content else None
        extracted_answers.append(number_match.group(1).zfill(len(answer)) if number_match else None)
    validity = torch.tensor([ans is not None for ans in extracted_answers], device="cpu", dtype=torch.bool)
    batched_total_rewards[validity == False] = float("nan")
    return batched_total_rewards, extracted_answers

def math_correctness(prm:str, tokenizer:AutoTokenizer, max_new_tokens:int, prompt:str, answer:str, device:torch.device, **kwargs) -> Callable[[torch.Tensor], Tuple[torch.Tensor, List]]:
    """
    A mathematical reasoning correctness reward function that employs a process reward model (PRM) to gauge the
    correctness of a full mathematical reasoning chain by multiplying the correctness estimate of each segment.

    Args:
        prm (str): The process-reward-model to be used.
        tokenizer (AutoTokenizer): The tokenizer associated with the generated tokens.
        max_new_tokens (int): The maximum new tokens the generator was allowed to produce. Used to identify invalid generations that were cut off.
        prompt (str): The mathematical question that the token sequence tries to answer.
        answer (str): The ground-truth response that the mathematical reasoning is supposed to arrive at.
        device (torch.device): The device to use for the process reward model (PRM) that is used to gauge the correctness of the reasoning.

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument and returns a correctness estimate by multiplying the PRM correctness probabilities of each step, as well as a list of final reasoning answer (put inside \\boxed{}).
    """
    
    return partial(_math_correctness, prm=prm, tokenizer=tokenizer, max_new_tokens=max_new_tokens, system_prompt="Please reason step by step, and put your final answer within \\boxed{}.", prompt=prompt, answer=answer, device=device)