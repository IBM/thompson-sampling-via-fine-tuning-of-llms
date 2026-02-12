import math, torch 
from typing import Callable, Tuple
from src.utils import misc
from src import gaussian_process
from copy import deepcopy

def iterate_through(gaussian_means:torch.Tensor, gaussian_stds:torch.Tensor, max_observation:float, t:int, delta:float) -> torch.Tensor:
    """Computes the iterate-through acquisition function that always picks the next element in the list

    Args:
        gaussian_means (torch.Tensor): the Gaussian process posterior means of shape (m,).
        gaussian_stds (torch.Tensor): the Gaussian process posterior standard deviations of shape (m,).
        t (int): time step (starts at 1)

    Returns:
        out (torch.Tensor): (m,) 1_{x=t-1} for x=0,...,m-1
    """
    return (torch.arange(gaussian_means.numel()) == t-1).to(torch.float)

def expected_improvement(gaussian_means:torch.Tensor, gaussian_stds:torch.Tensor, max_observation:float, t:int, delta:float) -> torch.Tensor:
    """Computes the expected improvement acquisition function

    Args:
        gaussian_means (torch.Tensor): the Gaussian process posterior means of shape (m,).
        gaussian_stds (torch.Tensor): the Gaussian process posterior standard deviations of shape (m,).
        max_observation (float): the best observation so far.

    Returns:
        out (torch.Tensor): (m,) E[max(0, F_x - max(observations))] for x=1,...,m
    """
    assert gaussian_means.ndim == 1 and gaussian_means.shape == gaussian_stds.shape, f"gaussian_means.shape = {gaussian_means.shape} and gaussian_stds.shape = {gaussian_stds.shape}"
    if not math.isfinite(max_observation): # possibly due to initialization in the first run
        return - torch.ones_like(gaussian_means) # uniform acquisition function indicating with -1 that the expected improvement cannot be computed
    deltas = (gaussian_means - max_observation)
    r = deltas * misc.Phi(deltas / gaussian_stds) + gaussian_stds * misc.phi(deltas / gaussian_stds)
    return r

def ucb(gaussian_means:torch.Tensor, gaussian_stds:torch.Tensor, max_observation:float, t:int, delta:float) -> torch.Tensor:
    """Computes the UCB acquisition function at time step t

    Args:
        gaussian_means (torch.Tensor): the Gaussian process posterior means of shape (m,).
        gaussian_stds (torch.Tensor): the Gaussian process posterior standard deviations of shape (m,).
        t (int): time step (starts at 1)
        delta (float): the accepted probability under which the GP-UCB regret bounds do not need to hold

    Returns:
        out (torch.Tensor): (m,) UCB(x) for x=1,...,m
    """
    assert gaussian_means.ndim == 1 and gaussian_means.shape == gaussian_stds.shape, f"gaussian_means.shape = {gaussian_means.shape} and gaussian_stds.shape = {gaussian_stds.shape}"
    beta = 2 * math.log(gaussian_means.numel() * t**2 * torch.pi**2 / (6*delta))
    return gaussian_means + beta**0.5 * gaussian_stds

def diagonal_thompson_sampling(gaussian_means: torch.Tensor, gaussian_stds:torch.Tensor, max_observation:float, t:int, delta:float) -> torch.Tensor:
    """Computes the Thompson sampling acquisition function assuming a diagonal covariance matrix

    Args:
        gaussian_means (torch.Tensor): the Gaussian process posterior means of shape (m,).
        gaussian_stds (torch.Tensor): the Gaussian process posterior standard deviations of shape (m,).

    Returns:
        out (torch.Tensor): (m,) A sample from the posterior N(gaussian_means, diag[gaussian_stds**2]) for x=1,...,m
    """
    return gaussian_means + torch.randn_like(gaussian_means) * gaussian_stds

def offline_bayesian_optimization(reward_model:gaussian_process.LinearGaussianProcess,
                                  reward_function: Callable,
                                  reward_for_invalid_generation: float,
                                  answer: str,
                                  X_rewards:torch.Tensor,
                                  X_validities:torch.Tensor,
                                  X_answers:torch.Tensor,
                                  X_tokens:torch.Tensor,
                                  X_kernel_features:torch.Tensor,
                                  num_bo_steps:int, 
                                  acquisition_function:Callable[[torch.Tensor, torch.Tensor, float, int, float], torch.Tensor],
                                  n_marginal_likelihood_warmup_steps:int=10, 
                                  ongoing_marginal_likelihood_maximization:bool=False,
                                  exploration_bonus:float=1.0,
                                  observe_invalid_generations:bool=False,
                                  log_hook:Callable[[float, float, float, float, float, float, float], None] = print) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Conducts Bayesian optimization on the given generations and returns the rewards over time, the validities over time, and the correctness over time.

    Args:
        reward_model (gaussian_process.LinearGaussianProcess): Gaussian reward model used to compute the posterior reward associated to generations.
        reward_function (Callable): Black-box function that takes a generated output (batched sequence of tokens in torch.Tensor format) and returns its associated reward, validity, and correctness.
        reward_for_invalid_generation (float): The reward given to invalid generations.
        answer (str): The answer that is searched in case of closed-ended generation, such as mathematical questions.
        X_rewards: The rewards associated to each generation.
        X_validities: The validity of each generation.
        X_answers: The answers of each generation in case it is closed-ended (like AIME2024).
        X_tokens: The tokens of each generation.
        X_kernel_features (torch.Tensor): The kernel features associated to each generation.
        num_bo_steps (int): Number of Bayesian optimization steps to perform.
        acquisition_function (Callable[[torch.Tensor, torch.Tensor, float, int, float], torch.Tensor]): The acquisition function that guides Bayesian optimization
        n_marginal_likelihood_warmup_steps (int, optional): the number of initial samples dedicated to finding good hyperparameters for the Gaussian prior without updating the reward model yet. Defaults to 10.
        ongoing_marginal_likelihood_maximization (bool, optional): Whether to keep using marginal likelihood maximization to update the prior parameters `nu` and `lambda` after the MLM warmup phase. Defaults to False.
        exploration_bonus (float, optional): An exploration bonus that primes Thompson sampling, UCB, etc. for additional exploration by rescaling the (MLM-fitted) Gaussian process amplitude `lambda` to `lambda` * `exploration_bonus`. 
            Note that black-box function agnostic Thompson sampling regret bounds (see https://arxiv.org/abs/1704.00445) and UCB regret bounds (see https://arxiv.org/abs/0912.3995) require a prior amplitude that scales with the complexity
            of the black-box function as measured by its RKHS norm, motivating the use of a larger amplitude than suggested by marginal likelihood maximization (MLM). Defaults to 1.0
        observe_invalid_generations (bool, optional): Whether to fit GP on invalid generations such as code that does not run or math proofs that did not terminate after `max_new_tokens` tokens. Defaults to False.
        log_hook (Callable[[float, float, float, float, float, float, float], None], optional): hook to log instantaneous reward, cumulative reward, highest reward, avg_top50_reward, BoN, majority, and w_BoN per step. Defaults to command line printing
    Returns:
        out (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing a tensor of query indices over time, a tensor of rewards over time, a tensor counting the total number of observations per index, a tensor of validity over time, and a tensor of correctness over time.
    """
    bo_indices = - torch.ones((num_bo_steps), dtype=torch.int64) # -1 is an invalid index, like NaN
    n_rewards_obs = torch.zeros((X_rewards.numel(),), dtype=torch.int64)
    bo_validities = torch.full((num_bo_steps,), 0.0)
    bo_answers = [None] * num_bo_steps
    highest_reward = -math.inf
    X_rewards_sum = torch.zeros_like(X_rewards, device="cpu") # accounts for re-evaluation of the black-box reward function
    # metrics
    bo_rewards = torch.full((num_bo_steps,), - math.inf)
    cumulative_rewards = torch.full((num_bo_steps,), - math.inf)
    simple_rewards = torch.full((num_bo_steps,), - math.inf)
    simple_rewards10 = torch.full((num_bo_steps,), - math.inf)
    BoNs = torch.full((num_bo_steps,), - math.inf)
    majoritys = torch.full((num_bo_steps,), - math.inf)
    w_BoNs = torch.full((num_bo_steps,), - math.inf)
    BoN10s = torch.full((num_bo_steps,), - math.inf)
    majority10s = torch.full((num_bo_steps,), - math.inf)
    w_BoN10s = torch.full((num_bo_steps,), - math.inf)
    pass_at_ks = torch.full((num_bo_steps,), - math.inf)
    for step in range(num_bo_steps):
        print(f"STEP {step}/{num_bo_steps}", end=", ")
        af = acquisition_function if step >= n_marginal_likelihood_warmup_steps else iterate_through # random evaluation of the samples = iterate-through evaluation of the samples (they are exchangeable a priori). Only after the warmup does the BO really start
        posterior_means = reward_model.posterior_mean(X_kernel_features)
        posterior_stds = exploration_bonus * (reward_model.posterior_var(X_kernel_features) ** .5)
        query = torch.argmax(af(posterior_means, posterior_stds, max_observation=highest_reward, t=step+1, delta=0.1)).item()
        print(f"query = {query}")
        bo_indices[step] = query
        if True: #n_rewards_obs[query] > 0: # query already occured => needs to re-evaluate the black-box reward function in case it is stochastic
            rwrd = reward_function(X_tokens[query:query+1, :])[0]
            rwrd[~rwrd.isfinite()] = reward_for_invalid_generation
            bo_rewards[step] = rwrd.item()
        else:
            bo_rewards[step] = X_rewards[query]
        X_rewards_sum[query] += bo_rewards[step]
        n_rewards_obs[query] += 1
        bo_validities[step] = float(X_validities[query].item())
        bo_answers[step] = X_answers[query]
        if observe_invalid_generations or bo_validities[step] == 1.0:
            reward_model.add_observation(X_kernel_features[query, :], bo_rewards[step], perform_marginal_likelihood_maximization = False)
        if ongoing_marginal_likelihood_maximization or step < n_marginal_likelihood_warmup_steps:
            reward_model.marginal_likelihood_maximization(min_obs=2)
        # recompute highest reward to ensure simple_reward is not fitting to noise
        if bo_rewards[step] < highest_reward:
            rwrd = reward_function(X_tokens[best_index:best_index+1, :])[0]
            rwrd[~rwrd.isfinite()] = reward_for_invalid_generation
            X_rewards_sum[best_index] += rwrd.item()
            n_rewards_obs[best_index] += 1
        # rewards
        avg_rewards = torch.nan_to_num(X_rewards_sum / n_rewards_obs.to(torch.float), nan=-math.inf) # evens out uncertainty in case of stochastic rewards, nan_to_num ensures that the average reward in the absence of observations is - infty
        best_index = torch.argmax(avg_rewards)
        top_10_indices = torch.topk(avg_rewards, k=10)[1]
        highest_reward = avg_rewards[best_index]
        avg_top10_reward = torch.mean(avg_rewards[top_10_indices])
        obs_answers, obs_rewards = [ans for idx, ans in enumerate(deepcopy(X_answers)) if n_rewards_obs[idx] > 0], [rew for idx, rew in enumerate(avg_rewards.clone()) if n_rewards_obs[idx] > 0]
        most_common_unweighted_strings = misc.most_common_string(obs_answers, k=10)[0]
        most_common_weighted_strings = misc.most_common_string(obs_answers, obs_rewards, k=10)[0]
        # pass@k
        pass_at_k = float(any(ans == answer for ans in obs_answers))
        # downstream accuracy metrics (top 1)
        BoN = float(answer == X_answers[best_index])
        majority = float(most_common_unweighted_strings[0] == answer)
        w_BoN = float(most_common_weighted_strings[0] == answer)
        # downstream accuracy metrics (average across top 10 to smooth out evaluation, corresponds to randomly picking one of the top 10)
        denominator = len(most_common_unweighted_strings)
        BoN10 = sum(float(answer == X_answers[top_10_indices[i].item()]) for i in range(denominator))/denominator
        majority10 = sum(float(most_common_unweighted_strings[i] == answer) for i in range(denominator))/denominator
        w_BoN10 = sum(float(most_common_weighted_strings[i] == answer) for i in range(denominator))/denominator
        # metrics
        cumulative_rewards[step] = torch.sum(bo_rewards[:step+1])
        simple_rewards[step] = highest_reward
        simple_rewards10[step] = avg_top10_reward
        BoNs[step] = BoN
        majoritys[step] = majority
        w_BoNs[step] = w_BoN
        BoN10s[step] = BoN10
        majority10s[step] = majority10
        w_BoN10s[step] = w_BoN10
        pass_at_ks[step] = pass_at_k
        # log instantaneous reward, cumulative reward, and highest reward per step
        log_hook(bo_rewards[step].item(), cumulative_rewards[step].item(), highest_reward, avg_top10_reward, BoN, majority, w_BoN, BoN10, majority10, w_BoN10, pass_at_k, bo_validities[step])
    return bo_indices, n_rewards_obs, avg_rewards, bo_validities, bo_answers, bo_rewards, cumulative_rewards, simple_rewards, simple_rewards10, BoNs, majoritys, w_BoNs, BoN10s, majority10s, w_BoN10s, pass_at_ks