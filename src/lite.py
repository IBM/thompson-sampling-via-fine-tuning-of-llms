from typing import Tuple, Callable
import torch
from src.utils import misc

def _find_normalizing_threshold(means:torch.Tensor, stds:torch.Tensor, epsilon:float) -> Tuple[float, float]:
    """
    Finds κ such that Σ_i P[N[i] >= κ] ≈ 1 for N[i] ~ N(means[i], stds[i]^2)

    Args:
        means (torch.Tensor): (m,)
        stds (torch.Tensor): (m,)
        epsilon (float): element-wise absolute convergence threshold 

    Returns:
        out (Tuple[float, float]): (kappa_lower_bound, kappa_upper_bound)
    """
    means = means.to(torch.float64)
    stds = stds.to(torch.float64)

    min_mu = torch.min(means)
    min_sigma = torch.min(stds)
    max_mu = torch.max(means)
    max_sigma = torch.max(stds)
    beta = misc.inv_Phi(1/torch.tensor(means.numel(), dtype=torch.float64)) # Inverse Normal CDF
    kappa_low = min_mu - beta * min_sigma
    kappa_up = max_mu - beta * max_sigma

    while(torch.max(misc.Phi((means - kappa_low) / stds) - misc.Phi((means - kappa_up) / stds)) >= epsilon): # unfortunately torch does not allow jitting while loops, so this Pytorch implementation is slower than the Jax version of LITE
        kappa = (kappa_low + kappa_up) / 2
        probs = misc.Phi((means - kappa) / stds)
        normalisation_delta = 1 - torch.sum(probs) # monotonously increasing in kappa
        kappa_low = torch.where(normalisation_delta < 0, kappa, kappa_low)
        kappa_up = torch.where(normalisation_delta >= 0, kappa, kappa_up)

    return kappa_low, kappa_up

def flite_pom(gaussian_means:torch.Tensor, gaussian_stds:torch.Tensor, epsilon:float=None) -> torch.Tensor:
    """
    Evaluates the F-LITE estimator of probability of maximality

    Args:
        gaussian_means (torch.Tensor): (m,)
        gaussian_stds (torch.Tensor): (m,)
        epsilon (float, optional): element-wise absolute convergence threshold, defaults to 1/(100·m)

    Returns:
        out (torch.Tensor): (m,) the probabilities of maximality
    """
    assert gaussian_means.ndim == 1 and gaussian_means.shape == gaussian_stds.shape, f"gaussian_means.shape = {gaussian_means.shape} and gaussian_stds.shape = {gaussian_stds.shape}"
    if epsilon is None:
        epsilon = 1/(100 * gaussian_means.numel()) 
    kappa_low, kappa_up = _find_normalizing_threshold(gaussian_means, gaussian_stds, epsilon)
    r_up = misc.Phi((gaussian_means - kappa_low) / gaussian_stds)
    r_low = misc.Phi((gaussian_means - kappa_up) / gaussian_stds)
    r = (r_up + r_low) / 2
    r /= r.sum() # evens out rounding errors
    return r


def log_flite_pom(gaussian_means:torch.Tensor, gaussian_stds:torch.Tensor, epsilon:float=None, base:float=10) -> torch.Tensor:
    """
    Evaluates the F-LITE estimator of probability of maximality

    Args:
        gaussian_means (torch.Tensor): (m,)
        gaussian_stds (torch.Tensor): (m,)
        epsilon (float, optional): element-wise absolute convergence threshold, defaults to 1/(100·m)
        base (float, optional): the base of the logarithm. Defaults to the natural logarithm.

    Returns:
        out (torch.Tensor): (m,) the logarithm of probabilities of maximality
    """
    assert gaussian_means.ndim == 1 and gaussian_means.shape == gaussian_stds.shape, f"gaussian_means.shape = {gaussian_means.shape} and gaussian_stds.shape = {gaussian_stds.shape}"
    if epsilon is None:
        epsilon = 1/(100 * gaussian_means.numel()) 
    kappa_low, kappa_up = _find_normalizing_threshold(gaussian_means, gaussian_stds, epsilon)
    log_r_up = misc.log_Phi((gaussian_means - kappa_low) / gaussian_stds)
    log_r_low = misc.log_Phi((gaussian_means - kappa_up) / gaussian_stds)
    log_corrective_factor = (1+torch.exp(log_r_low-log_r_up))/2
    log_r = log_r_up + torch.log(log_corrective_factor)
    log_r -= torch.logsumexp(log_r, dim=0) # evens out rounding errors
    return log_r / torch.log(torch.tensor(base))

def vapor_objective(means:torch.Tensor, stds:torch.Tensor, log_probs:torch.Tensor) -> float:
    """
    Computes the VAPOR objective function, see Tarbouriech et al. (https://arxiv.org/abs/2311.13294)

    Args:
        means (torch.Tensor): (m,)
        stds (torch.Tensor): (m,)
        log_probs (torch.Tensor): (m,)

    Returns:
        out (float): The VAPOR objective function value.
    """
    return torch.mean(means + stds * torch.sqrt(-2 * log_probs)).item()

def pom_objective_advantage_function(means:torch.Tensor, stds:torch.Tensor, log_probs:torch.Tensor, ref_log_probs:torch.Tensor, inverse_pom_activation_exp:Callable[[torch.Tensor], torch.Tensor], alpha:float) -> torch.Tensor:
    """
    Computes the variational F-LITE advantage function to be used in conjunction with the score trick for gradient ascent on the objective. 

    Args:
        means (torch.Tensor): (m,)
        stds (torch.Tensor): (m,)
        log_probs (torch.Tensor): (m,)
        ref_log_probs (torch.Tensor): (m,)
        inverse_pom_activation_exp (Callable[[torch.Tensor], torch.Tensor]): The inverse PoM activation function (in logarithmic units) to use. Allows switching between different estimators of PoM.
        alpha (float): The entropy regularization coefficient
    Returns:
        out (torch.Tensor): The advantage function. 
    """
    with torch.no_grad():
        Deltas = means - inverse_pom_activation_exp(log_probs) * stds - alpha * log_probs
        avg_Deltas = torch.mean(Deltas)
        advantage = (Deltas - avg_Deltas)
        std_advantage = (torch.mean((Deltas - avg_Deltas)**2))**.5 # uses that E[Advantage] = 0
        advantage /= (std_advantage+torch.finfo(std_advantage.dtype).eps)
    return advantage.detach()

def inv_vapor_exp(x: torch.Tensor) -> torch.Tensor:
    """
        Returns the inverse of the VAPOR activation function at e^x (i.e., v^{-1}(e^x))

        Args:
            x (torch.Tensor): The evaluation points. All points must be negative for well-definedness.

        Returns:
            out (Torch.tensor): The inverse of the VAPOR activation function evaluated element-wise at e^x
    """
    assert torch.all(x < 0), f"all entries of x must be negative for well-definedness of vapor_inv_cdf(e^x), but at least one entry is {x[(x>=0).nonzero()[0]]}"
    return 1/torch.sqrt(-2 * x) - torch.sqrt(-2 * x)

def inv_Phi_exp(x: torch.Tensor) -> torch.Tensor:
    """
        Returns the inverse of the cumulative distribution function of a standard normal at e^x (i.e., Phi^{-1}(e^x))

        Args:
            x (torch.Tensor): The evaluation points. All points must be negative for well-definedness.

        Returns:
            out (Torch.tensor): The inverse of the cumulative distribution function of a standard normal evaluated element-wise at e^x
    """
    assert torch.all(x < 0), f"all entries of x must be negative for well-definedness of Phi^{-1}(e^x), but at least one entry is {x[(x>=0).nonzero()[0]]}"
    exact = 2**0.5 * torch.erfinv(2*torch.exp(x)-1)
    stable = 1.2/(-x)**0.33 - (-2*x)**0.5
    # empirically, the stability of the inverse error function starts to deteriorate for x > -5, likewise, the stable approximation becomes exact for x < -5
    return torch.where(torch.logical_and(x > -5, exact.isfinite()), exact, stable)