from functools import cache
import torch, re
from transformers import AutoTokenizer
from datasets import load_dataset
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer

def shortness_of_response(tokenizer:AutoTokenizer, **kwargs) -> torch.Tensor:
    """
    A toy reward function that assigns a sequence of n tokens (non-padding) the score 1/(n/100+1).

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the tokens.

    Returns:
        out (Callable) A function that takes a torch.Tensor of batched tokens as an argument and returns the shortness score of each batch entry
    """
    def _shortness_of_response(X:torch.Tensor):
        eot = (X == tokenizer.eos_token_id) # check where <|endoftext|> occurs since responses are padded to equal length inside a batch
        response_length = ((eot.cumsum(dim=1) == 1) & eot).float().argmax(dim=1)
        return 1/(response_length/100+1), [None] * response_length.numel()
    return cache(_shortness_of_response)

def shakespearity(tokenizer:AutoTokenizer, **kwargs) -> torch.Tensor:
    """
    The Shakespearity reward function that assigns a sequence of tokens their shakespearity score, which we define
    as the type-token ratio times the fraction of words in the tiny_shakespeare dataset but not in the commonsense_qa dataset. 

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the tokens.

    Returns:
        out (Callable) A function that takes a torch.Tensor of batched tokens as an argument and returns the Shakespearity score of each batch entry
    """
    shakespeare_text = str(load_dataset('tiny_shakespeare')['train']['text'][0])
    shakespeare_words = set(re.sub(r'\s+', ' ', re.sub(r"[^\w\s']|[\n]", ' ', shakespeare_text)).lower().split()) # 11913 words
    comparison_text = " ".join(load_dataset("tau/commonsense_qa")["train"]["question"])
    comparison_words = set(re.sub(r'\s+', ' ', re.sub(r"[^\w\s']|[\n]", ' ', comparison_text)).lower().split()) # 8225 words
    shakespeare_words = shakespeare_words - comparison_words # 8511 words
    def _type_token_ratio(tokens:torch.Tensor, eos_token_id:int):
        assert tokens.ndim == 1, f"X has shape {tokens.shape}"
        assert not torch.is_floating_point(tokens), f"X has data type {tokens.dtype} instead of integer"
        tokens_no_eos = tokens[tokens != eos_token_id]
        return len(torch.unique(tokens_no_eos)) / len(tokens_no_eos) if len(tokens_no_eos) > 0 else 0.0
    def _shakespearity(X:torch.Tensor):
        "returns the fraction of words that overlaps with the tiny-shakespeare dataset multiplied by the type token ratio (TTR)"
        reward = []
        batched_generation = tokenizer.batch_decode(X, skip_special_tokens=True)
        for idx, generation in enumerate(batched_generation):
            words = re.sub(r'\s+', ' ', re.sub(r"[^\w\s']|[\n]", ' ', generation)).lower().split()        
            count = sum(1 for word in words if word in shakespeare_words)
            reward += [(count/len(words)) * _type_token_ratio(X[idx, :], tokenizer.eos_token_id) if len(words) > 0 else 0] ## ratio of words in tinyShakespeare but not in commonSenseQA times lexical type token ratio (TTR)
        return torch.tensor(reward, device="cpu", dtype=torch.double), [None] * len(reward)
    return cache(_shakespearity)

def faq(tokenizer:AutoTokenizer, device:torch.device, **kwargs) -> torch.Tensor:
    """
    The faq reward function that assigns a sequence of tokens their FAQ score, which we define as the BERT embedding similarity to an unknown, ideal, simplified FAQ.

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the tokens.

    Returns:
        out (Callable) A function that takes a torch.Tensor of batched tokens as an argument and returns the faq score of each batch entry
    """
    ideal_faq = """To reset your password, go to the login page and click the “Forgot Password” link. Enter your registered email address, and we'll send you instructions to create a new password. Make sure to check your spam folder if you don’t see the email within a few minutes."""
    embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device_map": "auto"}, tokenizer_kwargs={"padding_side": "left"}) #"attn_implementation": "flash_attention_2",
    ideal_faq_embedding = embedding_model.encode([ideal_faq], convert_to_tensor=True, normalize_embeddings=True).cpu()
    def _faq(X:torch.Tensor):
        "Returns the BERT similarity score to some unknown reference FAQ"
        completions = tokenizer.batch_decode(X, skip_special_tokens=True)
        embeddings = embedding_model.encode(completions, convert_to_tensor=True, normalize_embeddings=True).cpu()
        return torch.sum(embeddings * ideal_faq_embedding, dim=1), [None] * X.shape[0]
    return cache(_faq)