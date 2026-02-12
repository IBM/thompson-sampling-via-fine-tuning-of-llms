import wandb, torch, math, time
from src.utils import setup, misc
from scripts import helper
from src import baselines, language_model, kernel_features, classical_BO, gaussian_process, lite
from src.benchmarks import reward_functions

config = setup.parse_config(description="Run a Bayesian optimization (BO) filtering experiment, where the evaluation order is either as generated or informed by BO using a linear/rbf kernel in some LLM feature space.")
setup.init_torch(config['seed'])

# identify storage location
config['storage_id'] = config['storage_id'].removesuffix(".pt") if config.get('storage_id', None) is not None else f"{config['seed']}-{wandb.run.name}"
storage_path = f"{config['results_dir']}/{config['storage_id']}"

# Configure the prompt
tokenizer = language_model.get_tokenizer(config['tokenizer'])
total_prompt, prompt, answer = setup.process_prompt(config['seed'], tokenizer, config['prompt'], config['system_prompt'])
wandb.config['prompt_sample'] = config['prompt_sample'] = prompt

# generate or load token sequences, their generation probabilities, and associated feature embeddings from the generator
if not config.get('already_generated', False):
    # Configure the generator
    generator = language_model.get_model(config['generator']).eval()
    for param in generator.parameters():
        param.requires_grad = False
    print(f"Each LLM copy requires {misc.get_model_size(generator)} Gigabytes of memory to store the parameters (with type {list(generator.parameters())[0].dtype}).")
    generator = torch.compile(generator.to(setup.get_gpu(1)), dynamic=True)
    # sample from generator and extract generator embeddings
    X_tokens, X_log_prob, X_features_mte, X_features_mph, X_features_mlh, X_features_lph, X_features_llh = baselines.sample_and_extract_generator_features(total_prompt = total_prompt,
                                                                                                                                                           tokenizer = tokenizer,
                                                                                                                                                           generator = generator,
                                                                                                                                                           temperature = config['temperature'],
                                                                                                                                                           max_new_tokens = config['max_new_tokens'],
                                                                                                                                                           num_samples = config['num_samples'],
                                                                                                                                                           batch_size = config['batch_size'],
                                                                                                                                                           hidden_dim = config['hidden_dim'])
    torch.save({
    "config": config,
    "X_tokens": X_tokens,
    "X_log_prob": X_log_prob,
    "X_features_mte": X_features_mte,
    "X_features_mph": X_features_mph,
    "X_features_mlh": X_features_mlh,
    "X_features_lph": X_features_lph,
    "X_features_llh": X_features_llh, 
    }, storage_path + "-gen.pt")
    del generator # free up memory
    torch.cuda.empty_cache()
else:
    generation_data = torch.load(storage_path + "-gen.pt", map_location=setup.get_gpu(1))
    X_tokens, X_log_prob, X_features_mte, X_features_mph, X_features_mlh, X_features_lph, X_features_llh = generation_data["X_tokens"], generation_data["X_log_prob"], generation_data["X_features_mte"], generation_data["X_features_mph"], generation_data["X_features_mlh"], generation_data["X_features_lph"], generation_data["X_features_llh"]
    config['seed'] = generation_data['config']['seed'] # overwrites config seed to ensure that all randomness is equivalent to running the full script again, even in case generations (and evaluations) are loaded from old runs

# reseeds to ensure reproducibility even in case previous generations are re-used
setup.seed_randomness(config['seed'])
wandb.config.update({'seed': config['seed']}, allow_val_change=True)

# Evaluate the reward function or load the rewards
reward_function_template = getattr(reward_functions, config['reward_function'])
reward_function = reward_function_template(tokenizer=tokenizer, 
                                           max_new_tokens=config['max_new_tokens'], 
                                           prompt=prompt,
                                           answer=answer, 
                                           device=setup.get_gpu(1),
                                           prm=config.get('prm', None),
                                           llm=config.get('llm', None),
                                           judge_prompt=config.get('judge_prompt', None))
reward_for_invalid_generation = reward_functions.reward_for_invalid_generation[reward_function_template]

if not config.get('already_evaluated', False):
    X_rewards = torch.full((config['num_samples'],), float('nan'), device=setup.get_gpu(1), dtype=torch.float32)
    X_validities = torch.full((config['num_samples'],), False, device=setup.get_gpu(1), dtype=torch.bool)
    X_answers = [None] * config['num_samples']
    for batch_idx in range(math.ceil(config['num_samples']/config['batch_size'])):
        batch_left, batch_right = batch_idx * config['batch_size'], (batch_idx+1) * config['batch_size']
        new_X_tokens = X_tokens[batch_left:batch_right, :]
        start_time = time.perf_counter() 
        new_X_rewards, new_X_answers = reward_function(new_X_tokens)
        new_X_validities = new_X_rewards.isfinite()
        new_X_rewards[~new_X_validities] = reward_for_invalid_generation
        print(f"TIMING: {time.perf_counter() - start_time:.3f} seconds to evaluate a batch of size {config['batch_size']}") 
        print(f"Rewards: {new_X_rewards}", "Validities: ", new_X_validities, "Answers:", new_X_answers)
        X_rewards[batch_left:batch_right] = new_X_rewards
        X_validities[batch_left:batch_right] = new_X_validities
        X_answers[batch_left:batch_right] = new_X_answers
    X_rewards, X_validities, X_answers = X_rewards[:config['num_samples']], X_validities[:config['num_samples']], X_answers[:config['num_samples']]
    wandb.run.summary["best_generated"] = tokenizer.decode(X_tokens[torch.argmax(X_rewards), :], skip_special_tokens=True)
    torch.save({
    "config": config,
    "X_rewards": X_rewards,
    "X_validities": X_validities,
    "X_answers": X_answers,
    }, storage_path + "-eval.pt")
else:
    evaluation_data = torch.load(storage_path + "-eval.pt", map_location=setup.get_gpu(1))
    X_rewards, X_validities, X_answers = evaluation_data["X_rewards"], evaluation_data["X_validities"], evaluation_data["X_answers"]
    config['seed'] = evaluation_data['config']['seed'] # overwrites config seed to ensure that all randomness is equivalent to running the full script again, even in case generations (and evaluations) are loaded from old runs
assert X_rewards.numel() == X_validities.numel() == len(X_answers) == config['num_samples'], f"{X_rewards.numel()} {X_validities.numel()} {len(X_answers)} {config['num_samples']}"

# reseeds to ensure reproducibility even in case previous evaluations are re-used
setup.seed_randomness(config['seed'])
wandb.config.update({'seed': config['seed']}, allow_val_change=True)

# extract and transform features
feature_embedding_model, embedding_aggregation, kernel_feature_transformation, n_features = helper.feature_map(config)
if not isinstance(feature_embedding_model, int):
    prompt_ids = tokenizer(total_prompt, return_tensors="pt")['input_ids']
    X_kernel_features = feature_embedding_model(prompt_ids=prompt_ids, X_tokens=X_tokens, tokenizer=tokenizer)
    X_kernel_features = embedding_aggregation(X_kernel_features.to(setup.get_gpu(1)), X_tokens.to(setup.get_gpu(1)), tokenizer)
    feature_label = config['feature_embedding_model']
else:
    match feature_embedding_model:
        case 0:
            assert embedding_aggregation == kernel_features.sequence_mean, f'if feature_embedding_model is set to token embeddings, only the mean aggregation is supported'
            X_kernel_features = X_features_mte
            feature_label = "mte"
        case -2:
            if embedding_aggregation == kernel_features.sequence_mean:
               X_kernel_features = X_features_mph
               feature_label = "mph"
            elif embedding_aggregation == kernel_features.sequence_latest:
                X_kernel_features = X_features_lph 
                feature_label = "lph"
            else:
                assert False, f"embedding aggregation {embedding_aggregation} is not supported"
        case -1:
            if embedding_aggregation == kernel_features.sequence_mean:
               X_kernel_features = X_features_mlh
               feature_label = "mlh"
            elif embedding_aggregation == kernel_features.sequence_latest:
                X_kernel_features = X_features_llh 
                feature_label = "llh"
            else:
                assert False, f"embedding aggregation {embedding_aggregation} is not supported" 
        case _:
            assert False, f"extraction of embedding from layer {feature_embedding_model} of the generator is not supported"

# whitening is not yet implemented for tosfit but only for the baselines
if config.get('whitening', False):
    # centering
    mean_features = X_kernel_features.mean(dim=0, keepdim=True)
    X_kernel_features = X_kernel_features - mean_features

    # ZCA whitening matrix
    U, S, Vt = torch.linalg.svd(X_kernel_features, full_matrices=False)
    print("singular values before whitening: ", S)
    epsilon = 1e-10
    S_inv = 1.0 / torch.clamp(S, min=epsilon)
    whitening_matrix = Vt.T @ torch.diag(S_inv) @ Vt  * (X_kernel_features.shape[0] - 1)
    torch.save({
    "features_singular_values": S,
    "features_mean": mean_features,
    "whitening_matrix": whitening_matrix,
    }, storage_path + "-whitening.pt")

    X_kernel_features = X_kernel_features @ whitening_matrix

assert torch.isfinite(X_kernel_features).all(), f"AW tensor of shape {X_kernel_features.shape} and data type {X_kernel_features.dtype} has only {torch.isfinite(X_kernel_features).sum().item()} finite entries"

X_kernel_features = kernel_feature_transformation(X_kernel_features)
cosine_similarity_matrix = X_kernel_features @ X_kernel_features.t()
cosim_counts, cosim_bins = torch.histogram(cosine_similarity_matrix.cpu().detach().flatten(), 100)
print("feature cosine similarity matrix histogram", (cosim_counts, cosim_bins))
wandb.log({"cosim matrix": wandb.Histogram(np_histogram=(cosim_counts.numpy(), cosim_bins.numpy()))})

# Configure the reward model
reward_model = gaussian_process.LinearGaussianProcess(n_features=n_features, 
                                                      nar=config['nar'],
                                                      device=setup.get_gpu(1), 
                                                      dtype=torch.float64)
print(f"The GP reward model uses {reward_model.memory / 1024**3} Gigabytes of memory.")

# Define the acquisition function to be used
match config['acquisition_function']:
    case "IT":
        acquisition_function = classical_BO.iterate_through
    case "EI":
        acquisition_function = classical_BO.expected_improvement
    case "UCB":
        acquisition_function = classical_BO.ucb
    case "TS":
        acquisition_function = classical_BO.diagonal_thompson_sampling
    case _:
        assert False, f"The acquisition function {config['acquisition_function']} is not implemented."

# perform Bayesian optimization
if answer is None:
    log_hook = lambda inst_reward, cum_reward, simp_reward, avg_top10_reward, BoN, majority, w_BoN, BoN10, majority10, w_BoN10, pass_at_k, validities: wandb.log({"inst_reward": inst_reward, 
                                                                                                                                                  "cumulative_reward": cum_reward, 
                                                                                                                                                  "simple_reward": simp_reward, 
                                                                                                                                                  "avg_top10_simple_reward": avg_top10_reward, 
                                                                                                                                                  "validities": validities})
else:
    log_hook = lambda inst_reward, cum_reward, simp_reward, avg_top10_reward, BoN, majority, w_BoN, BoN10, majority10, w_BoN10, pass_at_k, validities: wandb.log({"inst_reward": inst_reward, 
                                                                                                                                                  "cumulative_reward": cum_reward, 
                                                                                                                                                  "simple_reward": simp_reward, 
                                                                                                                                                  "avg_top10_simple_reward": avg_top10_reward, 
                                                                                                                                                  "BoN": BoN, 
                                                                                                                                                  "majority": majority, 
                                                                                                                                                  "w_BoN": w_BoN,
                                                                                                                                                  "avg_top10_BoN": BoN10, 
                                                                                                                                                  "avg_top10_majority": majority10,
                                                                                                                                                  "avg_top10_w_BoN": w_BoN10,
                                                                                                                                                  "pass_at_k": pass_at_k,
                                                                                                                                                  "validities": validities})
bo_indices, n_rewards_obs, avg_rewards, validities, answers, instant_rewards, cum_rewards, simple_rewards, \
simple_rewards10, BoNs, majoritys, w_BoNs, BoN10s, majority10s, w_BoN10s, pass_at_ks = classical_BO.offline_bayesian_optimization(reward_model=reward_model,
                                                                                                                                  reward_function=reward_function,
                                                                                                                                  reward_for_invalid_generation=reward_for_invalid_generation,
                                                                                                                                  answer=answer,
                                                                                                                                  X_rewards = X_rewards,
                                                                                                                                  X_validities = X_validities,
                                                                                                                                  X_answers = X_answers,
                                                                                                                                  X_kernel_features = X_kernel_features,
                                                                                                                                  X_tokens = X_tokens,
                                                                                                                                  num_bo_steps = config['num_samples'],
                                                                                                                                  acquisition_function = acquisition_function,
                                                                                                                                  n_marginal_likelihood_warmup_steps = config['n_marginal_likelihood_warmup_steps'],
                                                                                                                                  ongoing_marginal_likelihood_maximization = config['ongoing_marginal_likelihood_maximization'],
                                                                                                                                  exploration_bonus = config['exploration_bonus'],
                                                                                                                                  observe_invalid_generations = config['observe_invalid_generations'],
                                                                                                                                  log_hook=log_hook)

torch.save({"config": config,
            "simple_reward": simple_rewards, 
            "simple_reward_top10": simple_rewards10,
            "cum_reward": cum_rewards,
            "BoN": BoNs,
            "BoN_top10": BoN10s,
            "majority": majoritys,
            "majority_top10": majority10s,
            "w_BoN": w_BoNs,
            "w_BoN_top10": w_BoN10s,
            "pass_at_k": pass_at_ks,
            "query_indices": bo_indices,
            "rewards": instant_rewards,
            "n_rewards_obs": n_rewards_obs,
            "validities": validities,
            "answers": answers}, storage_path + "-" + feature_label + "-" + config['kernel_feature_transformation'] + "-" + config['acquisition_function'] + '-' + str(config['exploration_bonus']) +"-metrics.pt")  
wandb.finish()