from functools import cache
import torch, re
from transformers import AutoTokenizer
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def protein_stability(tokenizer:AutoTokenizer, **kwarg):
    """
    Returns a thermal stability reward function that estimates the thermal stability of the provided amino acid sequence (in token form), 
    based on the negative thermal instability index (Guruprasad K., Reddy B.V.B., Pandit M.W. Protein Engineering 4:155-161(1990).)

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the generation tokens (amino acid sequence).

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument, encoding an amino acid sequence, and returns a thermal stability estimate of the sequence.
    """
    def _protein_stability(X:torch.Tensor):
        rewards = []
        batched_generation = tokenizer.batch_decode(X, skip_special_tokens=True)
        for generation in batched_generation:
            cleaned_generation = re.sub(r"[^a-zA-Z]", "", generation) # removes white spaces and \n etc.
            analysis = ProteinAnalysis(cleaned_generation)
            try:
                stability = -analysis.instability_index()
            except:
                stability = float('nan')
            rewards += [stability]
        return torch.tensor(rewards, device="cpu", dtype=torch.double), [None] * len(rewards)
    return cache(_protein_stability)