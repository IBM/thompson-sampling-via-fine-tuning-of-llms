import multiprocessing, torch, math, time
import numpy as np
np.seterr(all='ignore')
from transformers import AutoTokenizer
from src.benchmarks.utils import safely_run_python
from src import language_model

# must be globally defined due to pickling from starmap from the multiprocessing module
def _run_bin_packing(gen_func_body:str, INITIAL_BIN_CAPACITY:int, MAX_ITEM_SIZE:int, item_sizes:np.ndarray, PRINT:bool):
        NUMBER_OF_ITEMS = item_sizes.size
        func_code = "def heuristic(item_size: float, bin_capacities: np.ndarray) -> np.ndarray:\n    scores = np.ones_like(bin_capacities)\n" + gen_func_body + "\n    return scores"
        item_sizes = np.minimum(np.ceil(item_sizes).astype(int), np.array(MAX_ITEM_SIZE, dtype=int)) # caps and discretizes to integers
        bins_lower_bound = np.sum(item_sizes).item() / INITIAL_BIN_CAPACITY # integer by integer division results in a float
        bin_capacities = np.ones((NUMBER_OF_ITEMS, ), dtype=int) * INITIAL_BIN_CAPACITY
        valid = True
        exception = None
        try:
            heuristic = safely_run_python.compile_restricted_function(func_code, "heuristic", {'np': np, 'numpy': np, 'math': math, 'njit': njit})
            safely_run_python.execute_restricted_function(heuristic, {'item_size': float(item_sizes[0].item()), 'bin_capacities': bin_capacities.flatten().astype(float)}, max_seconds=30) # ensures any jit compilatiosn are carried through in case they are used
            for step in range(NUMBER_OF_ITEMS):
                item_size = item_sizes[step].item()
                assert math.isfinite(item_size) and np.isfinite(bin_capacities).all(), f"item_size is {item_size} and bin_capacities is {bin_capacities}"
                feasible_bin_indices = np.argwhere(bin_capacities >= item_size)
                priority_scores = safely_run_python.execute_restricted_function(heuristic, {'item_size': float(item_size), 'bin_capacities': bin_capacities[feasible_bin_indices].flatten().astype(float)}, max_seconds=60/NUMBER_OF_ITEMS) # allow at most 1 minute for all iterations to avoid infinite loops
                picked_bin = feasible_bin_indices[np.argmax(priority_scores)]
                bin_capacities[picked_bin] -= item_size
            bins_used = np.sum(bin_capacities != INITIAL_BIN_CAPACITY)
        except Exception as e:
            valid = False
            bins_used = NUMBER_OF_ITEMS # either the syntax of the code was invalid or it timed out, in either case we penalize this as using all the bins available
            exception = e
        finally:
            if PRINT:
                print("EVALUATION of an LLM solution\n", func_code)
                if valid == False:
                    print("EXCEPTION in LLM evaluation:", exception, flush=True)
            excess_bins = bins_used / bins_lower_bound  - 1.0
            return excess_bins, valid
            
def bin_packing_excess_bins(tokenizer:AutoTokenizer, **kwargs) -> torch.Tensor:
    """
    Returns the average excess bins (AEB) reward function for bin-packing that simulates a bin-packing instance and stochastically 
    evaluates the heuristic encoded by the passed token torch.Tensor.

    Args:
        tokenizer (AutoTokenizer): The tokenizer associated with the tokens.

    Returns:
        out (Callable) A function that takes a torch.Tensor of tokens as an argument and returns the empirical AEB score. Invalid code is assigned an NaN reward.
    """
    def _bin_packing_excess_bins(X:torch.Tensor):
        NUMBER_OF_ITEMS = 500 #0 # set to 500 for faster iteration, but should ultimately be increased
        NUMBER_OF_EXPERIMENTS = 50
        INITIAL_BIN_CAPACITY = 150
        MAX_ITEM_SIZE = 100
        WEIBULL_PARAMETERS = (45, 3)
        item_sizes = torch.distributions.weibull.Weibull(*WEIBULL_PARAMETERS).sample((NUMBER_OF_EXPERIMENTS, X.shape[0], NUMBER_OF_ITEMS)).numpy(force=True)
        with multiprocessing.Pool() as pool:
            results = pool.starmap(_run_bin_packing, [(safely_run_python.extract_code(gen_func_body), INITIAL_BIN_CAPACITY, MAX_ITEM_SIZE, item_sizes[e, b, :], e==0) for b, gen_func_body in enumerate(tokenizer.batch_decode(X, skip_special_tokens=True)) for e in range(NUMBER_OF_EXPERIMENTS)]) # second loop is "inner loop"
        regret, regret_validity = zip(*results)
        regret, regret_validity = list(regret), list(regret_validity)
        regret = [sum(regret[i*NUMBER_OF_EXPERIMENTS+j] for j in range(NUMBER_OF_EXPERIMENTS)) / NUMBER_OF_EXPERIMENTS for i in range(X.shape[0])]
        regret_validity = [all(regret_validity[i*NUMBER_OF_EXPERIMENTS+j] for j in range(NUMBER_OF_EXPERIMENTS)) for i in range(X.shape[0])]
        regret = [reg if regret_validity[idx] else float("nan") for idx, reg in enumerate(regret)]
        return -torch.tensor(list(regret), device="cpu", dtype=torch.double), [None] * regret.numel()
    return _bin_packing_excess_bins

if __name__ == "__main__":
    fun_search_bin_packing_heuristic = """\
    scores = 1000 * np.ones(bin_capacities.shape)
    # Penalize bins with large capacities.
    scores -= bin_capacities * (bin_capacities-item_size)
    # Extract index of bin with best fit.
    index = np.argmin(bin_capacities)
    # Scale score of best fit bin by item size.
    scores[index] *= item_size
    # Penalize best fit bin if fit is not tight.
    scores[index] -= (bin_capacities[index] - item_size)**4"""

    first_fit_bin_packing_heuristic = """\
    scores = np.ones(bin_capacities.shape)
    """

    best_fit_bin_packing_heuristic = """\
    scores = - bin_capacities
    """

    EoC_bin_packing_heuristic = """\
    scores = np.log(item_size) * (bin_capacities ** 2) / (item_size * np.sqrt(bin_capacities - item_size)) + (bin_capacities / item_size) ** 3
    scores[bin_capacities == bin_capacities.max()] = -np.inf
    """

    EoH_bin_packing_heuristic = """\
    diff = bin_capacities - item_size
    exp = np.exp(diff)
    sqrt = np.sqrt(diff)
    ulti = 1 - diff / bin_capacities 
    comb = ulti * sqrt
    adjust = np.where(diff > (item_size * 3), comb + 0.8, comb + 0.3)
    hybrid_exp = bin_capacities / ((exp + 0.7) * exp)
    scores = hybrid_exp + adjust
    """

    best_own_heuristic = """\
    scores = np.zeros(bin_capacities.shape[0], dtype = float)
    # Compute the ratio between the bin capacity and item size
    ratio = bin_capacities / item_size
    
    # Assign scores to bin for priority to choose
    for index, ratio in enumerate(ratio):
        if ratio >= 2: ### (own comment) among all bins that have twice the capacity needed, go with first fit
            scores[index] = 1
        elif 1 <= ratio < 2: ### (own comment) among all bins that have less than twice the capacity, go with best fit
            scores[index] = 2 - ratio
        else: ### (own comment) bins that don’t fit are excluded
            scores[index] = 0
    ### perfect fit gets as good a score as bins with large capacity, which is a bit odd
    """

    tokenizer = language_model.get_tokenizer("Qwen/Qwen2.5-Coder-0.5B")

    tokenized_prompts = tokenizer([fun_search_bin_packing_heuristic, 
                                EoC_bin_packing_heuristic,
                                EoH_bin_packing_heuristic,
                                first_fit_bin_packing_heuristic,
                                best_fit_bin_packing_heuristic,
                                best_own_heuristic], padding=True, return_tensors="pt")['input_ids']
    start_time = time.perf_counter()
    regret, _ = bin_packing_excess_bins(tokenized_prompts, tokenizer)
    print("TIME TAKEN FOR EVALUATION", time.perf_counter() - start_time)

    print("FUNSEARCH", regret[0].item())
    print("EoC", regret[1].item())
    print("EoH", regret[2].item())
    print("FIRST_FIT", regret[3].item())
    print("BEST_FIT", regret[4].item())
    print("BEST_OWN", regret[5].item())