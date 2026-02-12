from typing import Callable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src import language_model
from src.benchmarks.math.reward import _math_correctness

def aime2024_prompt_optimization(llm:str, prm:str, tokenizer:AutoTokenizer, device:torch.device, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    A prompt reward function that runs an LLM on AIME2024 with different system prompts to find optimal prompts for mathematical reasoning.

    Args:
        llm (str): The large language model to be used on AIME2024.
        tokenizer (AutoTokenizer): The tokenizer associated with the generated system prompt tokens.
        device (torch.device): The device to use for running the experiment.

    Returns:
        out (Callable) A function that takes a torch.Tensor of system prompt tokens as an argument and returns the success rate of running an LLM with the provided system prompt on AIME2024.
    """
    expensive_llm = AutoModelForCausalLM.from_pretrained(llm, cache_dir="./model_weights/", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device).eval() 
    expensive_llm_tokenizer = AutoTokenizer.from_pretrained(llm, cache_dir="./model_weights/", use_fast=False)
    aime2024 = load_dataset("HuggingFaceH4/aime_2024", split="train") # only has train split with 30 entries
    sample_questions = False

    def _aime2024_prompt_optimization(X:torch.Tensor):
        batch_size = X.shape[0]
        batched_system_prompts = tokenizer.batch_decode(X, skip_special_tokens=True)
        batched_rewards, batched_answers = [0.0]*batch_size, [0.0]*batch_size
        for batch, system_prompt in enumerate(batched_system_prompts): 
            if sample_questions:
                sampled_question = 3 #torch.randint(0, 30, ()).item()
            for question_idx, question in enumerate(aime2024):
                if sample_questions and sampled_question != question_idx:
                    continue
                prompt = question['problem']
                solution = question['solution']
                answer = question['answer']
                data = {
                        "system": system_prompt,
                        "query": prompt,
                    }
                messages = [
                    {"role": "system", "content": data['system']},
                    {"role": "user", "content": data['query']},
                ]
                total_prompt_ids = expensive_llm_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True,
                    return_tensors='pt'
                )
                X, _ = language_model.generate_tokens_exluding_backprop(total_prompt_ids, expensive_llm, expensive_llm_tokenizer, temperature=0.7, batch_size=1, max_new_tokens=8192)

                rew, answer = _math_correctness(X, prm, expensive_llm_tokenizer, 8192, system_prompt, prompt, answer, device)

                batched_rewards[batch] += rew[0] if rew[0].isfinite() else 0.0 # the reward of invalid answers is NaN, so we need to filter these out
                batched_answers[batch] += rew[0]
            batched_rewards[batch] /= (30.0 if not sample_questions else 1.0)
        return torch.tensor(batched_rewards, device="cpu", dtype=torch.float), [None] * len(batched_rewards)
        
    return _aime2024_prompt_optimization

def generate_and_judge(llm:str, judge_prompt:str, tokenizer:AutoTokenizer, device:torch.device, **kwargs):
    """
    Returns a generate & judge reward function that employs an LLM both to generate a response based on the 
    prompt provided in token form as well as an LLM as a judge to provide feedback on the generation.

    Args:
        llm (str): The LLM that is used to generate based on the prompt as well as to judge the generation.
        judge_prompt (str): The prompt that conditions the llm to act as a judge.
        tokenizer (AutoTokenizer): The tokenizer associated with the provided tokens (generated prompt).
        device (torch.device): The device to use for the llm (as a judge).

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument (prompt) and returns the scores (LLM as a judge) of the freshly generated text.
    """
    expensive_llm = torch.compile(AutoModelForCausalLM.from_pretrained(llm, cache_dir="./model_weights/", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)).eval() 
    expensive_llm_tokenizer = AutoTokenizer.from_pretrained(llm, cache_dir="./model_weights/", use_fast=False)

    def _extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
        try:
            if split_str in answer:
                rating = answer.split(split_str)[1]
            else:
                rating = answer
            digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)] #regex expression matches all numbers either with or without decimal point.
            return max(0.0, min(10.0, float(digit_groups[0]))) # just extracts first number provided as a rating in case of ambiguity when multiple ratings are provided.
        except Exception as e:
            print(e)
            return 0.0
    
    def _generate_and_judge(X:torch.Tensor):
        batched_prompts = tokenizer.batch_decode(X, skip_special_tokens=True)
        judges_scores = []
        for prompt in batched_prompts:
            data = {
                    "system": 'You are a helpful assistant. You do as told without explaining yourself or asking for clarification.',
                    "query": prompt,
                }
            messages = [
                {"role": "system", "content": data['system']},
                {"role": "user", "content": data['query']},
            ]
            total_prompt_ids = expensive_llm_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
            X, _ = language_model.generate_tokens_exluding_backprop(total_prompt_ids, expensive_llm, expensive_llm_tokenizer, temperature=1.0, batch_size=1, max_new_tokens=1024)
            generated_text = expensive_llm_tokenizer.decode(X[0])
            # LLM as a judge 
            data = {
                    "system": 'You are a helpful assistant. You do as told without explaining yourself or asking for clarification.',
                    "query": judge_prompt.format(generated_text=generated_text),
                }
            messages = [
                {"role": "system", "content": data['system']},
                {"role": "user", "content": data['query']},
            ]
            total_prompt_ids = expensive_llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors='pt'
            )
            X, _ = language_model.generate_tokens_exluding_backprop(total_prompt_ids, expensive_llm, expensive_llm_tokenizer, temperature=1.0, batch_size=1, max_new_tokens=4096)
            judgement = expensive_llm_tokenizer.decode(X[0])
            print(judgement)
            judges_scores.append(_extract_judge_score(judgement))
        return torch.tensor(judges_scores, device="cpu"), [None] * len(judges_scores)
    return _generate_and_judge