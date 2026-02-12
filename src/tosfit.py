from typing import Callable, Tuple, Union
import wandb, time, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import gaussian_process, language_model, lite
from src.utils import misc

def _fit_pom_with_variational_flite(reward_model:gaussian_process.LinearGaussianProcess, 
                                    feature_embedding_model:Union[Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor], int], 
                                    embedding_aggregation:Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor], 
                                    kernel_feature_transformation:Callable[[torch.Tensor], torch.Tensor], 
                                    inverse_pom_activation_exp: Callable[[torch.Tensor], torch.Tensor], 
                                    exploration_bonus: float,
                                    prompt_ids:torch.Tensor, 
                                    tokenizer:AutoTokenizer, 
                                    generator:AutoModelForCausalLM, 
                                    reference_generator: AutoModelForCausalLM,
                                    temperature:float, 
                                    max_new_tokens:int, 
                                    batch_size:int, 
                                    mini_batch_size:int,
                                    alpha:float,
                                    disable_gradient_ascent:bool=False,
                                    remove_exploration:bool=False) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float, float]:
    """
        Conducts gradient ascent on the regularized variational LITE objective using the score trick and variance-standardized advantage functions (as in GRPO we do not rescale the KL-regularization term).
        IMPORTANT: It is assumed that the optimizer is tied to the parameters through hooks that get activated during the backward pass (which significantly saves on memory compared to full back-propagation followed by a gradient step).

        Args:
            reward_model (gaussian_process.LinearGaussianProcess): Gaussian reward model used to compute the posterior.
            feature_embedding_model (Union[Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor], int]): A function that takes positional arguments `prompt_ids, X_tokens, tokenizer` and returns a per-token embedding. If set to an integer, the corresponding hidden layer from the `reference_generator` is taken as embedding.
            embedding_aggregation (Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor]): The embedding aggregation method that takes positional arguments `token-wise embeddings, X_tokens, tokenizer` and returns an aggregated embedding.
            kernel_feature_transformation (Callable[[torch.Tensor], torch.Tensor]): The transformation method that takes the argument `embedding` and returns a transformed embedding.
            inverse_pom_activation_exp (Callable[[torch.Tensor], torch.Tensor]): The inverse of the PoM activation function that is used to estimate PoM, evaluated at exp(input) such that log probabilities can be passed.
            A kernel rescaling function that takes the log probability of prior LLM generation as an argument. The function is multiplicatively applied to rescale the final kernel feature vector.
            exploration_bonus (float): An exploration bonus that primes Thompson sampling for additional exploration by rescaling the (MLM-fitted) Gaussian process amplitude `lambda` to `lambda` * `exploration_bonus`. 
                Note that black-box function agnostic Thompson sampling regret bounds (see https://arxiv.org/abs/1704.00445) require a prior amplitude that scales with the complexity of the black-box function as measured by its RKHS norm,
                motivating the use of a larger amplitude than suggested by marginal likelihood maximization (MLM).
            prompt_ids (torch.Tensor): Tensor of prompt tokens that induce the autoregressive sampling process.
            tokenizer (AutoTokenizer): Tokenizer that maps from text to tokens and vice-versa.
            generator (AutoModelForCausalLM): Fine-tuned language model used as the policy.
            reference_generator (AutoModelForCausalLM): Pre-trained language model used for regularization (and possibly as an embedding model)
            temperature (float): Temperature parameter for stochastic sampling from the generator.
            max_new_tokens (int): Maximum length of generated responses.
            batch_size (int): Batch size for finetuning. Note that variance-standardized advantage functions (as in GRPO) are used and hence the batch elements act as baselines for each other. The batch size determines the number of samples for GRPO.
            mini_batch_size (int): Mini batch size for finetuning, where the batch is split into mini-batches of size `mini_batch_size` which are consecutively backpropagated through and used to update the parameters. Note that all mini-batches share the same (old) policy. Small mini-batch size helps with memory issues.
            alpha (float): Coefficient of the entropy regularization term.
            disable_gradient_ascent (bool, optional): If set to True, disables gradient ascent and thus renders this function pure. Defaults to False.

        Returns:
            out (Tuple[torch.Tensor, torch.Tensor, float, float, float, float, float, float]): The generated tokens, the kernel embeddings associated with the generated tokens, the loss from the VAPOR (Tarbouriech et al. 2024) objective, the KL-divergence used for regularization, 
            the entropy of the generator, the time spent on autoregressive LLM generation, the time spent on the GP reward model, the total time spent
    """
    feature_layer = feature_embedding_model if isinstance(feature_embedding_model, int) else None # None indicates that we are not interested in extracting any hidden layers from the reference_generator
    start_time = time.perf_counter()
    with torch.no_grad():
        # generate tokens
        X_tokens, X_log_probs = language_model.generate_tokens_exluding_backprop(prompt_ids, generator, tokenizer, temperature, batch_size, max_new_tokens)
        time_spent_on_LLM_GEN = time.perf_counter() - start_time
        # extract reference log probabilities and token-wise kernel embedding
        X_ref_log_probs, X_kernel_embedding = language_model.low_memory_log_probs_and_hidden_states(mini_batch_size, prompt_ids, X_tokens, reference_generator, tokenizer, feature_layer, temperature, max_new_tokens)
        if feature_layer is None:
            X_kernel_embedding = feature_embedding_model(prompt_ids=prompt_ids, X_tokens=X_tokens, tokenizer=tokenizer)
        X_kernel_embedding = kernel_feature_transformation(embedding_aggregation(X_kernel_embedding.to(reward_model.device), X_tokens.to(reward_model.device), tokenizer)) # features on sphere
        # compute reward posterior
        start_time2 = time.perf_counter()
        posterior_means = reward_model.posterior_mean(X_kernel_embedding)
        posterior_stds = exploration_bonus * reward_model.posterior_var(X_kernel_embedding) ** .5
        time_spent_on_GP = time.perf_counter() - start_time2
        print("\\mu(x)", posterior_means, "\\sigma(x)", posterior_stds, "\\v^{-1}(p_\\theta(x))", inverse_pom_activation_exp(X_ref_log_probs)) 
        advantage = lite.pom_objective_advantage_function(posterior_means.to(generator.device), posterior_stds.to(generator.device), X_log_probs, X_ref_log_probs.to(generator.device), inverse_pom_activation_exp, alpha)
        # record statistics
        kl_divergence = misc.kl_divergence_estimator(X_log_probs.to(X_ref_log_probs.device), X_ref_log_probs)
        entropy = misc.entropy_estimator(X_log_probs)
        loss = -lite.vapor_objective(posterior_means, posterior_stds, X_log_probs.to(posterior_means.device)) # as a surrogate for the variational F-LITE objective, which is numerically less stable to compute (but whose derivatives can be computed stably)
        # move data to generator GPU
        X_ref_log_probs, posterior_means, posterior_stds = X_ref_log_probs.to(generator.device), posterior_means.to(generator.device), posterior_stds.to(generator.device)
    if not disable_gradient_ascent:
            # splitting AutoGrad computation into mini-batches to bring down the memory significantly (the auto-regressive token generation is anyhow much more sequential than this final forward/backward pass)
            for mini_batch_idx in range(math.ceil(batch_size/mini_batch_size)):
                mini_batch_left, mini_batch_right = mini_batch_idx * mini_batch_size, (mini_batch_idx+1) * mini_batch_size
                mini_batch_X_tokens, mini_batch_advantage = X_tokens[mini_batch_left:mini_batch_right, :], advantage[mini_batch_left:mini_batch_right]
                # forward pass to build up AutoGrad graph for the objective function
                mini_batch_X_log_probs, _ = language_model.log_probs_and_hidden_states(prompt_ids, mini_batch_X_tokens, generator, tokenizer, None, temperature, max_new_tokens)
                objective = torch.sum(mini_batch_advantage.detach() * mini_batch_X_log_probs) / batch_size # does not correct for gradients already applied in this batch, because due to rounding error there is no reliable estimator of the generation probability
                objective.backward()
    total_time_spent = time.perf_counter() - start_time 
    return X_tokens, X_kernel_embedding, loss, kl_divergence, entropy, time_spent_on_LLM_GEN, time_spent_on_GP, total_time_spent

def tosfit(reward_function:Callable[[torch.Tensor], torch.Tensor], 
         reward_for_invalid_generation:float,
         answer:str,
         reward_model:gaussian_process.LinearGaussianProcess, 
         feature_embedding_model:Union[Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor], int], 
         embedding_aggregation:Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor], 
         kernel_feature_transformation:Callable[[torch.Tensor], torch.Tensor], 
         inverse_pom_activation_exp: Callable[[torch.Tensor], torch.Tensor], 
         total_prompt:str,
         tokenizer:AutoTokenizer, 
         generator:AutoModelForCausalLM,
         reference_generator:AutoModelForCausalLM,
         temperature:float, 
         max_new_tokens:int, 
         optimizer_constructor:torch.optim.Optimizer, 
         num_samples:int, 
         bo_batch_size:int,
         fine_tune_steps_per_bo_step:int, 
         batch_size:int, 
         mini_batch_size:int,
         alpha:float, 
         observe_invalid_generations:bool = False,
         n_marginal_likelihood_warmup_steps:int = 10,
         ongoing_marginal_likelihood_maximization:bool = False,
         exploration_bonus:float = 1.0):
    """
    Runs Thompson sampling through fine-tuning (tosfit). tosfit balances exploration 
    (trying new actions) and exploitation (choosing promising actions with known rewards) 
    to optimize a black-box 'reward' function. It uses a Bayesian approach with a GP 
    reward model whose kernel is derived from an embedding model. The generative model
    is finetuned to the posterior probability of maximality of the reward model. This ensures that sampling 
    thereof corresponds to Thompson sampling and avoids intractable maximization of acquisition functions.

    Args:
        reward_function (Callable): Black-box function that takes a generated output (batched sequence of tokens in torch.Tensor format) and returns its associated reward, validity, and correctness.
        reward_for_invalid_generation (float): The reward given to invalid generations.
        answer (str): The answer that is searched in case of closed-ended generation, such as mathematical questions.
        reward_model (gaussian_process.LinearGaussianProcess): Gaussian reward model used to compute the posterior reward associated to generations.
        feature_embedding_model (Union[Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor], int]): A function that takes positional arguments `prompt_ids, X_tokens, tokenizer` and returns a per-token embedding. If set to an integer, the corresponding hidden layer from the `reference_generator` is taken as embedding.
        embedding_aggregation (Callable[[torch.Tensor, torch.Tensor, AutoTokenizer], torch.Tensor]): The embedding aggregation method that takes positional arguments `token-wise embeddings, X_tokens, tokenizer` and returns an aggregated embedding.
        kernel_feature_transformation (Callable[[torch.Tensor], torch.Tensor]): The transformation method that takes the argument `embedding` and returns a transformed embedding.
        inverse_pom_activation_exp (Callable[[torch.Tensor], torch.Tensor]): The inverse of the PoM activation function that is used to estimate PoM, evaluated at exp(input) such that log probabilities can be passed.
        total_prompt (str): The prompt that is used to induce the sampling process. For instruction-tuned models this includes the system prompt.
        tokenizer (AutoTokenizer): Tokenizer that maps from text to tokens and vice-versa. Should match the `generator` and `reference_generator`.
        generator (AutoModelForCausalLM): Pre-trained language model used as the policy from which we sample. Axiomatically, it encodes the prior probability of maximality.
        reference_generator (AutoModelForCausalLM): Pre-trained language model used for regularization (and possibly as an embedding model). Should be an independent copy of `generator`.
        temperature (float): Temperature parameter for stochastic sampling from the generator.
        max_new_tokens (int): Maximum length of LLM generation.
        optimizer_constructor (torch.optim.Optimizer): Constructor that takes in a list of parameters and returns an optimizer tied to those parameters (configured for objective maximization). Used during call of gradient hooks in backpropagation for low-memory gradient ascent.
        num_samples (int): Number of Bayesian optimization steps to perform.
        bo_batch_size (int): The batch size of Bayesian optimization, i.e., how many candidates are simultaneously evaluated.
        fine_tune_steps_per_bo_step (int): Number of batches to process in each Bayesian optimization step for finetuning to updated reward model. Linearly affects the computational cost of tosfit.
        batch_size (int): Batch size for finetuning. Note that variance-standardized advantage functions (as in GRPO) are used and hence the elements of a batch act as baselines for each other.  The batch size determines the number of samples for GRPO.
        mini_batch_size (int): Mini batch size for finetuning, where the batch is split into mini-batches of size `mini_batch_size` which are consecutively backpropagated through and used to update the parameters. Note that all mini-batches share the same (old) policy. Small mini-batch size helps with memory issues.
        alpha (float): Coefficient of the entropy regularization term.
        observe_invalid_generations (bool, optional): Whether to train on invalid generations such as code that does not run or math proofs that did not terminate after `max_new_tokens` tokens. Defaults to False.
        n_marginal_likelihood_warmup_steps (int, optional): The number of initial samples dedicated to finding good hyperparameters for the Gaussian prior without updating the model yet. Defaults to 10.
        ongoing_marginal_likelihood_maximization (bool, optional): Whether to keep using marginal likelihood maximization to update the prior parameters `nu` and `lambda` after the MLM warmup phase. Defaults to False.
        exploration_bonus (float, optional): An exploration bonus that primes Thompson sampling for additional exploration by rescaling the (MLM-fitted) Gaussian process amplitude `lambda` to `lambda` * `exploration_bonus`. 
            Note that black-box function agnostic Thompson sampling regret bounds (see https://arxiv.org/abs/1704.00445) require a prior amplitude that scales with the complexity of the black-box function as measured by its RKHS norm,
            motivating the use of a larger amplitude than suggested by marginal likelihood maximization (MLM). Defaults to 1.0
    
    Notes:
        This implementation integrates with Weights & Biases (wandb) for monitoring of progress.
    """ 
    n_batched_steps = math.ceil(num_samples / bo_batch_size)
    num_samples = bo_batch_size * n_batched_steps
    # generations & their scores
    generations = torch.full((num_samples, max_new_tokens), tokenizer.eos_token_id)
    rewards = torch.full((num_samples,), - math.inf)
    n_rewards_obs = torch.zeros((num_samples,), dtype=torch.int) # best observation so far gets resampled in case of stochasticity and the average is taken to avoid overfitting to noise.
    validities = torch.full((num_samples,), 0.0)
    answers = [None] * num_samples
    # metrics
    simple_reward = torch.zeros(n_batched_steps, dtype=torch.float64)
    cumulative_reward = torch.zeros(n_batched_steps, dtype=torch.float64)
    simple_reward_top10 = torch.zeros(n_batched_steps, dtype=torch.float64)
    BoN = torch.zeros(n_batched_steps, dtype=torch.float64)
    majority = torch.zeros(n_batched_steps, dtype=torch.float64) 
    w_BoN = torch.zeros(n_batched_steps, dtype=torch.float64)
    BoN_top10 = torch.zeros(n_batched_steps, dtype=torch.float64)
    majority_top10 = torch.zeros(n_batched_steps, dtype=torch.float64) 
    w_BoN_top10 = torch.zeros(n_batched_steps, dtype=torch.float64)
    pass_at_k = torch.zeros(n_batched_steps, dtype=torch.float64)
    # statistics
    entropy = torch.zeros(n_batched_steps, dtype=torch.float64) 
    kl_divergence = torch.zeros(n_batched_steps, dtype=torch.float64)
    loss = torch.zeros(n_batched_steps, dtype=torch.float64)
    time_spent_on_LLM = 0
    time_spent_on_LLM_GEN = 0
    time_spent_on_GP = 0
    time_spent_on_EVAL = 0
    # run Bayesian optimization
    misc.enable_direct_grad(generator.parameters(), optimizer_constructor)
    prompt_ids = tokenizer(total_prompt, return_tensors="pt")['input_ids']
    for bo_step in range(0, num_samples, bo_batch_size):
        print(f"Bayesian optimization step {bo_step}/{num_samples}")
        # consistency updates that align the LLM with the posterior probability of maximality induced from the posterior Gaussian reward model
        for _ in range(fine_tune_steps_per_bo_step):
            X_tokens, X_kernel_embedding, step_loss, step_kl_divergence, step_entropy, autoregressive_time, gp_time, total_time = _fit_pom_with_variational_flite(reward_model, 
                                                                                                                                                         feature_embedding_model, 
                                                                                                                                                         embedding_aggregation, 
                                                                                                                                                         kernel_feature_transformation,
                                                                                                                                                         inverse_pom_activation_exp,
                                                                                                                                                         exploration_bonus,
                                                                                                                                                         prompt_ids, 
                                                                                                                                                         tokenizer, 
                                                                                                                                                         generator, 
                                                                                                                                                         reference_generator,
                                                                                                                                                         temperature, 
                                                                                                                                                         max_new_tokens,
                                                                                                                                                         batch_size,
                                                                                                                                                         mini_batch_size,
                                                                                                                                                         alpha,
                                                                                                                                                         disable_gradient_ascent=bo_step < n_marginal_likelihood_warmup_steps)
            entropy[bo_step//bo_batch_size] += step_entropy
            kl_divergence[bo_step//bo_batch_size] += step_kl_divergence
            loss[bo_step//bo_batch_size] += step_loss
            time_spent_on_LLM += total_time
            time_spent_on_LLM_GEN += autoregressive_time
            time_spent_on_GP += gp_time
        entropy[bo_step//bo_batch_size] /= fine_tune_steps_per_bo_step
        kl_divergence[bo_step//bo_batch_size] /= fine_tune_steps_per_bo_step
        loss[bo_step//bo_batch_size] /= fine_tune_steps_per_bo_step
        # select candidates for evaluation
        X_tokens = X_tokens[:bo_batch_size]
        X_kernel_embedding = X_kernel_embedding[:bo_batch_size, :]
        # obtain observations using the last batch of LLM generations
        start_time = time.perf_counter()
        X_reward, X_answers = reward_function(X_tokens)
        X_validity = X_reward.isfinite()
        X_reward[~X_validity] = reward_for_invalid_generation
        time_spent_on_EVAL += time.perf_counter() - start_time
        print("The instantaneous rewards are ", X_reward, " with validity ", X_validity, " and correctness ", X_answers == answer)
        # update reward model
        Y_observed = torch.ones_like(X_validity, dtype=torch.bool) if observe_invalid_generations else X_validity
        if Y_observed.any():
            start_time = time.perf_counter()
            reward_model.add_observations(X_kernel_embedding[Y_observed, :], X_reward[Y_observed], ongoing_marginal_likelihood_maximization if bo_step >= n_marginal_likelihood_warmup_steps else True, min_obs=n_marginal_likelihood_warmup_steps)
            time_spent_on_GP += time.perf_counter() - start_time
        # record candidates & their scores
        generations[bo_step:bo_step+bo_batch_size, 0:X_tokens.shape[1]] = X_tokens
        rewards[bo_step:bo_step+bo_batch_size] = X_reward
        n_rewards_obs[bo_step:bo_step+bo_batch_size] += 1
        validities[bo_step:bo_step+bo_batch_size] = X_validity.type(torch.float)
        answers[bo_step:bo_step+bo_batch_size] = X_answers
        # resample best solution in case of stochastic rewards to ensure that we do not overfit to statistical anomaly
        best_index = rewards.argmax().item()
        if X_reward.max() < rewards[best_index]:
            best_reward, _ = reward_function(generations[best_index].unsqueeze(0))
            best_reward[~best_reward.isfinite()] = reward_for_invalid_generation
            rewards[best_index] = (rewards[best_index]*n_rewards_obs[best_index]+best_reward) / (n_rewards_obs[best_index]+1)
            n_rewards_obs[best_index] += 1
        # update metrics
        most_common = misc.most_common_string(answers[:bo_step+bo_batch_size], k=10)[0]
        most_common_weighted = misc.most_common_string(answers[:bo_step+bo_batch_size], rewards[:bo_step+bo_batch_size], k=10)[0]
        denominator = len(most_common)
        best_indices = torch.topk(rewards, k=10)[1]
        simple_reward[bo_step//bo_batch_size] = rewards[best_index]
        simple_reward_top10[bo_step//bo_batch_size] = torch.mean(rewards[best_indices])
        cumulative_reward[bo_step//bo_batch_size] = cumulative_reward[max(0, bo_step//bo_batch_size-1)] + torch.sum(X_reward)
        BoN[bo_step//bo_batch_size] = float(answers[best_index] == answer if answers[best_index] else False)
        BoN_top10[bo_step//bo_batch_size] = sum(float(answers[best_indices[i]] == answer if answers[best_indices[i]] else False) for i in range(denominator))/denominator
        majority[bo_step//bo_batch_size] = float(most_common[0] == answer)
        majority_top10[bo_step//bo_batch_size] = sum(float(most_common[i] == answer) for i in range(denominator))/denominator
        w_BoN[bo_step//bo_batch_size] = float(most_common_weighted[0] == answer)
        w_BoN_top10[bo_step//bo_batch_size] = sum(float(most_common_weighted[i] == answer) for i in range(denominator))/denominator
        pass_at_k[bo_step//bo_batch_size] = float(any(ans == answer for ans in answers[:bo_step+bo_batch_size]))

        time_spent_on_LLM_per_step = time_spent_on_LLM / (bo_step+bo_batch_size)
        time_spent_on_LLM_GEN_per_step = time_spent_on_LLM_GEN / (bo_step+bo_batch_size)
        time_spent_on_GP_per_step = time_spent_on_GP / (bo_step+bo_batch_size)
        time_spent_on_EVAL_per_step = time_spent_on_EVAL / (bo_step+bo_batch_size)
        # update weights and biases
        wandb.run.summary["best_generated"] = generations[best_index]
        wandb.run.summary["best_generated_str"] = tokenizer.decode(generations[best_index], skip_special_tokens=True)
        wandb.run.summary["time_LLM/BO_step"] = time_spent_on_LLM_per_step
        wandb.run.summary["time_LLM_GEN/BO_step"] = time_spent_on_LLM_GEN_per_step
        wandb.run.summary["time_GP/BO_step"] = time_spent_on_GP_per_step
        wandb.run.summary["time_EVAL/BO_step"] = time_spent_on_EVAL_per_step
        if answer is None:
            wandb.log({"inst_reward": torch.mean(X_reward).item(), # instantaneous reward
                    "simple_reward": simple_reward[bo_step//bo_batch_size].item(), 
                    "cumulative_reward": cumulative_reward[bo_step//bo_batch_size].item(),
                    "avg_top10_simple_reward": simple_reward_top10[bo_step//bo_batch_size].item(),

                    "validities": (torch.sum(X_validity == True) / X_validity.numel()).item(), 
                    "kl_divergence": kl_divergence[bo_step//bo_batch_size].item(), 
                    "generator_entropy": entropy[bo_step//bo_batch_size].item(),
                    "vapor_loss": loss[bo_step//bo_batch_size].item()}, # we report on the VAPOR loss instead of F-LITE, since it is easier to compute stably. This stands in contrast to the gradients of F-LITE, which can be stably computed.
                    step=bo_step+bo_batch_size)
        else:
            wandb.log({"inst_reward": torch.mean(X_reward).item(), # instantaneous reward
                    "simple_reward": simple_reward[bo_step//bo_batch_size].item(), 
                    "cumulative_reward": cumulative_reward[bo_step//bo_batch_size].item(),
                    "avg_top10_simple_reward": simple_reward_top10[bo_step//bo_batch_size].item(),
                    
                    "BoN": BoN[bo_step//bo_batch_size].item(),
                    "avg_top10_BoN": BoN_top10[bo_step//bo_batch_size].item(),
                    "majority": majority[bo_step//bo_batch_size].item(),
                    "avg_top10_majority": majority_top10[bo_step//bo_batch_size].item(),
                    "w_BoN": w_BoN[bo_step//bo_batch_size].item(),
                    "avg_top10_w_BoN": w_BoN_top10[bo_step//bo_batch_size].item(),
                    "pass_at_k": pass_at_k[bo_step//bo_batch_size].item(),

                    "validities": (torch.sum(X_validity == True) / X_validity.numel()).item(), 
                    "kl_divergence": kl_divergence[bo_step//bo_batch_size].item(), 
                    "generator_entropy": entropy[bo_step//bo_batch_size].item(),
                    "vapor_loss": loss[bo_step//bo_batch_size].item()}, # we report on the VAPOR loss instead of F-LITE, since it is easier to compute stably. This stands in contrast to the gradients of F-LITE, which can be stably computed.
                    step=bo_step+bo_batch_size)
    print(f"DONE")
    return simple_reward, cumulative_reward, simple_reward_top10, BoN, majority, w_BoN,\
           BoN_top10, majority_top10, w_BoN_top10, pass_at_k, entropy, kl_divergence, loss,\
           time_spent_on_LLM_per_step, time_spent_on_LLM_GEN_per_step,\
           time_spent_on_GP_per_step, time_spent_on_EVAL_per_step,\
           generations, rewards, n_rewards_obs, validities, answers