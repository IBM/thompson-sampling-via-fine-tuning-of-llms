from typing import List, Union, Tuple, List, Iterable, Callable
from copy import deepcopy
import torch, itertools

def norm_hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281 ^ ord(ch)*997 ) & 0xFFFFFFFF
  return hash

def phi(x: torch.Tensor) -> torch.Tensor:
    """
        Returns the probability density function of a standard normal (i.e., phi(x))

        Args:
            x (torch.Tensor): The evaluation points.
        
        Returns:
            out (Torch.tensor): The probability density function of a standard normal evaluated element-wise at x.
    """
    return torch.exp(-x**2 / 2) / (2 * torch.pi)**.5

def Phi(x: torch.Tensor) -> torch.Tensor:
    """
        Returns the cumulative distribution function of a standard normal (i.e., Phi(x))

        Args:
            x (torch.Tensor): The evaluation points.

        Returns:
            out (Torch.tensor): The cumulative distribution function of a standard normal evaluated element-wise at x.
    """
    return 0.5 + 0.5 * torch.erf(x / 2 ** .5)

def log_Phi(x: torch.Tensor, 
            base: float = None) -> torch.Tensor:
    """
        Returns the logarithm of the cumulative distribution function of a standard normal (i.e., log Phi(x))

        Args:
            x (torch.Tensor): The evaluation points.
            base (float, optional): The base of the logarithm. Defaults to Euler's constant (corresponding to the natural logarithm)

        Returns:
            out (Torch.tensor): The logarithm of the cumulative distribution function of a standard normal evaluated element-wise at x.
    """
    if base is None:
        return torch.special.log_ndtr(x)
    else:
        return torch.special.log_ndtr(x) / torch.log(torch.tensor(base))

def inv_Phi(x: torch.Tensor) -> torch.Tensor:
    """
        Returns the inverse of the cumulative distribution function of a standard normal (i.e., Phi^{-1}(x))

        Args:
            x (torch.Tensor): The evaluation points. All points must lie in the interval (0,1) for well-definedness.

        Returns:
            out (Torch.tensor): The inverse of the cumulative distribution function of a standard normal evaluated element-wise at x
    """
    assert torch.all(x > 0) and torch.all(x < 1), f"all entries of x must be in (0,1) for well-definedness of Phi^{-1}(x), but at least one entry is {x[(x<=0 or x>=1).nonzero()[0]]}"
    return 2**0.5 * torch.erfinv(2*x-1)

def vapor_cdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(torch.sqrt(x**2 + 4) - x)**2 / 8)

def vapor_inv_cdf(x: torch.Tensor) -> torch.Tensor:
    assert torch.all(x > 0) and torch.all(x < 1), f"all entries of x must be in (0,1) for well-definedness of the inverse (vapor) cdf, but at least one entry is {x[(x<=0 or x>=1).nonzero()[0]]}" 
    return 1/torch.sqrt(-2 * torch.ln(x)) - torch.sqrt(-2 * torch.ln(x))

def geometric_mean(input:torch.Tensor, 
                   dim:Union[int, Tuple[int, ...]]=None, 
                   keepdim:bool=False, 
                   dtype:torch.dtype=None, 
                   out:torch.Tensor=None, 
                   ignore_nans:bool=False) -> torch.Tensor:
    """
        Returns the geometric mean of each row of the `input` tensor in the given dimension `dim`. If `dim` is a list of dimensions, reduce over all of them.

        If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1.
        Otherwise, `dim` is squeezed, resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s).

        Args:
            input (torch.Tensor): The input tensor
            dim (int or tuple of ints, optional): The dimension or dimensions to reduce. If `dim` is None, all dimensions are reduced. Defaults to None.
            keepdim (bool, optional): Whether the output tensor has `dim` retained or not. Defaults to False.
            dtype (torch.dtype, optional): The desired data type of returned tensor.
                If specified, the input tensor is casted to dtype before the operation
                is performed. This is useful for preventing data type overflows. Default to None.
            out (torch.Tensor, optional): The output tensor. Must have the right dimension. Defaults to None.
            ignore_nans (bool, optional): Whether to ignore nans in ln(input) for the calculation of the geometric mean. Defaults to False.

        Returns:
            out (Torch.tensor): The inverse of the cumulative distribution function of a standard normal evaluated element-wise at e^x
    """
    log_x = torch.log(input)
    log_geometric_mean = torch.nanmean(log_x, dim, keepdim=keepdim, dtype=dtype) if ignore_nans else torch.mean(log_x, dim, keepdim=keepdim, dtype=dtype)
    result = torch.exp(log_geometric_mean)
    if out is not None:
        assert out.shape == result.shape, f"Shape mismatch: result.shape is {result.shape} but out.shape is {out.shape}"
        out[:]=result
    return result

def kl_divergence_estimator(log_p:torch.Tensor, 
                            log_q:torch.Tensor) -> float:
    """
    An unbiased estimator of the Kullback-Leibler divergence E_{x~p} [log p(x) - log q(x)].
    Makes use of the variance reduction technique from J. Schulman, "Approximating kl divergence", 2020. URL http://joschu.net/blog/kl-approx.html. 

    Args:
        log_p (torch.Tensor): Samples from log p(x) where x ~ p
        log_q (torch.Tensor): Samples from log q(x) where x ~ p

    Returns: 
        out (float): an unbiased estimator of Kullback-Leibler divergence

    """
    return log_p.mean().item() - log_q.mean().item() + (torch.exp(log_q - log_p).mean() - 1).item()

def entropy_estimator(log_p:torch.Tensor) -> float:
    """
    An unbiased estimator of the entropy -E_{x~p} [log p(x)].

    Args:
        log_p (torch.Tensor): Samples from log p(x) where x ~ p

    Returns:
        out (float): an unbiased estimator of entropy

    """
    return - log_p.mean().item()

def random_fourier_features(X:torch.Tensor, 
                            D:int, 
                            ls:float, 
                            random_normals:torch.Tensor=None) -> torch.Tensor:
    """
    Generate random Fourier features to approximate the RBF kernel exp(-||x-z||^2 / (2ls^2)). 
    Uses low-variance RFFs according to https://arxiv.org/pdf/1506.02785
    
    Args:
        X (torch.Tensor): Input tensor of shape (batch, n_features)
        D (int): Number of random Fourier features, must be a multiple of two
        ls (float): Length scale parameter for the RBF kernel
        random_normals (torch.Tensor, optional): A sample from torch.randn(n_features, D//2) that is used for consistent random fourier features.
            If set to None, a new sample is drawn. Defaults to None.
    
    Returns:
        out (torch.Tensor): Feature matrix after transformation of shape (batch, D)
    """
    assert D%2==0, f'D is {D} but must be even'
    batch_size, n_features = X.shape
    if random_normals is None:
        random_normals = torch.randn(n_features, D//2, device=X.device, dtype=X.dtype)
    else:
        assert random_normals.shape == (n_features, D//2), f"random_normals.shape = {random_normals.shape}"
    W = random_normals / ls
    z = torch.cat([torch.sin(X @ W), torch.cos(X @ W)], dim=1) * (2/D)**.5
    return z

def get_optimizer_params(optimizer:torch.optim.Optimizer) -> List[torch.nn.Parameter]:
    """
    Returns a list of all the parameters that are optimized by `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose parameters we seek.

    Returns:
        out (List[torch.nn.Parameter]): A list of all the parameters that are handled by the optimizer.
    """
    return list(itertools.chain.from_iterable(group['params'] for group in optimizer.param_groups))

def get_gradient_norm(optimizer:torch.optim.Optimizer, 
                      p_norm:float=2.0) -> float:
    """
    Returns the gradient norm of the parameters that are optimized by `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose parameters' gradient norm we seek.
        p_norm (float, optional): The p-norm to use. p=1.0 corresponds to the Manhattan norm, and p=2.0 is the Euclidean norm. Defaults to 2.0.
    Returns:
        out (float): A list of all the parameters that are handled by the optimizer.
    """
    total_norm = 0.0
    with torch.no_grad():
        for param in get_optimizer_params(optimizer):
            if param.grad is not None:  # Ensure parameter has a gradient
                param_norm = torch.norm(param.grad.detach(), p_norm)
                total_norm += param_norm.item() ** p_norm
    return (total_norm ** (1.0 / p_norm))

def enable_direct_grad(parameters:Iterable[torch.nn.Parameter],
                                    optimizer_constructor:Callable) -> None:
    """
    For each parameter, creates an optimizer and hooks it to the parameter during backpropagation to directly update model weights without storing gradients.
    This avoids the memory overhead of first computing the full gradient vector and then applying the update.
    
    Args:
        parameters (Iterable[torch.nn.Parameter]): Parameters that will automatically be updated by their optimizers during backpropagation.
        optimizer_constructor (Callable): Constructor that takes a list of parameters and creates an optimizer linked to them. 
    """
    optimizer_dict = {p: optimizer_constructor([p]) for p in parameters if p.requires_grad}
    # Define our hook, which will call the optimizer ``step()`` and ``zero_grad()``. Together with the optimizer_dict, this reduces model memory by a factor of 3 
    # because gradients are directly applied and freed during backpropagation (which requires twice #param memory unless the foreach option is set to false)
    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad(set_to_none=True)
    for p in optimizer_dict.keys():
        p.register_post_accumulate_grad_hook(optimizer_hook)

def get_model_size(model:torch.nn.Module) -> float:
    """
        Returns the total model size (excluding activations) in Gigabytes.

        Args:
            model (torch.nn.Module): The model whose size we are interested in.

        Results:
            out (int): The number of Gigabytes the module occupies (not counting activations).
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_gb = (param_size + buffer_size) / 1024**3
    return size_all_gb

def most_common_string(lst:List[str], weights:List[float]=None, k:int=1):
    """
    Finds the k most frequently occurring non-None strings in a weighted list and returns
    the strings along with the indices of their first occurrence.

    Parameters:
        lst (List[str]): A list of strings where some elements may be None.
        weights (List[float], optional): A list of weights to use. Defaults to uniform weighting
        k (int, optional): How many strings to find. Defaults to 1.
    Returns:
        tuple: A tuple (List[string], List[index]) where the list with 'string' is the k most frequent 
               non-None strings in the list, and the list with 'index' is the k corresponding indices of their 
               first occurrence. In case there are too many None entries, the lists are filled up with None's and their positions.
    """
    lst = deepcopy(lst)
    weights = [1.0]*len(lst) if weights is None else deepcopy(weights)
    assert len(lst) == len(weights)
    counts = {}
    none_indices = []
    for i, item in enumerate(lst):
        if item is not None:
            if item in counts:
                counts[item]['count'] += weights[i]
            else:
                counts[item] = {'count': weights[i], 'index': i}
        else:
            none_indices.append(i)

    top_k = sorted(counts.items(), key=lambda x: x[1]['count'], reverse=True)[:k]
    if len(top_k) < k:
        top_k += [(None, {'count':1, 'index':idx}) for idx in none_indices][:k - len(top_k)]
    return [el[0] for el in top_k], [el[1]['index'] for el in top_k]