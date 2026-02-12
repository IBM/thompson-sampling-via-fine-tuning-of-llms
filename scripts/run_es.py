import wandb, torch, math, time, re, random
from src.utils import setup, misc
from src import language_model
from src.benchmarks import reward_functions

def tournament_selection(population: list, k: int) -> dict:
    """
    Selects a single individual from the population using K-Tournament Selection. If `K` exceeds the population size, the fittest entry of the whole population is returned.

    Args:
        population: A list of dictionaries, where each dict has a 'score' key.
                    Example: [{'id': 1, 'score': 0.8}, {'id': 2, 'score': 0.9}]
        k: The size of the tournament (number of individuals to sample).
           A larger k means stronger selection (more exploitation).

    Returns:
        The dictionary representing the fittest individual from the tournament.
    """
    if not population:
        raise ValueError("Population cannot be empty.")
    if k <= 0:
        raise ValueError(f"Tournament size 'k' is {k} but must be positive.")
    if k < len(population):
        # 1. Randomly select k individuals (with replacement is standard)
        tournament_members = random.choices(population, k=k)
    else:
        tournament_members = population
    # 2. Find the individual with the highest score (fittest)
    fittest_individual = None
    best_score = -float('inf')
    for individual in tournament_members:
        # Assuming higher scores are better (higher fitness)
        if individual['score'] > best_score:
            best_score = individual['score']
            fittest_individual = individual
    # 3. Return the fittest individual from the tournament
    return fittest_individual

def token_mutation(sequence: str, vocabulary: list, substitution_rate: float = 0.05, indel_rate: float = 0.01) -> str:
    """
    Performs random point mutations on a sequence based on the mutation rate.
    
    Args:
        sequence: The input amino acid sequence string.
        substitution_rate: The probability that any single token will be substituted.
        indel_rate: The probability that any single position will experience an insertion or deletion (total indel rate is 2x this value).

    Returns:
        The mutated sequence string.
    """
    mutated_seq = list(sequence)
    i = 0
    while i < len(mutated_seq):
        if random.random() < substitution_rate:
            mutated_seq[i] = random.choice(vocabulary)
        if random.random() < indel_rate:
            mutated_seq.insert(i, random.choice(vocabulary))
            i += 1
        if random.random() < indel_rate:
            del mutated_seq[i]
            i -= 1
        i += 1
    return "".join(mutated_seq)

def single_point_crossover(parent1: str, parent2: str) -> str:
    """
    Performs a single-point crossover between two parent sequences.
    
    Args:
        parent1: The first parent sequence string.
        parent2: The second parent sequence string.

    Returns:
        The offspring sequence string.
    """
    min_len = min(len(parent1), len(parent2))
    crossover_point = random.randint(0, min_len)
    return parent1[:crossover_point] + parent2[crossover_point:]

config = setup.parse_config(description="Run Optimization using Evolutionary Search.")
setup.init_torch(config['seed'])

# Configure the tokenizer and generator
tokenizer = language_model.get_tokenizer(config['tokenizer'])
generator = language_model.get_model(config['generator']).eval()
print(f"The LLM requires {misc.get_model_size(generator)} Gigabytes of memory to store the parameters (with type {list(generator.parameters())[0].dtype}).")
generator = torch.compile(generator.to(setup.get_gpu(1)), dynamic=True)

# Define the external reward function
reward_function_template = getattr(reward_functions, config['reward_function'])
reward_function = reward_function_template(tokenizer=tokenizer, 
                                           max_new_tokens=config['max_new_tokens'], 
                                           prompt=config['prompt'],
                                           answer=None, 
                                           device=setup.get_gpu(4),
                                           prm=None,
                                           llm=None,
                                           judge_prompt=None)
reward_for_invalid_generation = reward_functions.reward_for_invalid_generation[reward_function_template]

# generations & their scores
generations = torch.full((config['num_samples'], config['max_new_tokens']), tokenizer.eos_token_id)
rewards = torch.full((config['num_samples'],), - math.inf)
n_rewards_obs = torch.zeros((config['num_samples'],), dtype=torch.int) # best observation so far gets resampled in case of stochasticity and the average is taken to avoid overfitting to noise.
validities = torch.full((config['num_samples'],), 0.0)

# metrics
batch_size = config.get('bo_batch_size', 1)
n_batched_steps = math.ceil(config['num_samples'] / batch_size)
simple_reward = torch.zeros(n_batched_steps, dtype=torch.float64)
cumulative_reward = torch.zeros(n_batched_steps, dtype=torch.float64)
simple_reward_top10 = torch.zeros(n_batched_steps, dtype=torch.float64)

# statistics
time_spent_on_LLM_GEN = 0
time_spent_on_EVAL = 0

population = [{'candidate': config['prompt'], 'score': reward_function(tokenizer(config['prompt'], return_tensors="pt")['input_ids'])[0].item()}]
for step in range(0, config['num_samples'], batch_size):
    print(f"EVO step {step}/{config['num_samples']}")
    dynamic_batch_size = batch_size if step+batch_size < config['num_samples'] else config['num_samples'] - step
    # run EVO
    parents = [(tournament_selection(population, k=config['tournament_size'])['candidate'], tournament_selection(population, k=config['tournament_size'])['candidate']) for _ in range(dynamic_batch_size)]
    if config['operator'] == 'LLM':
        batch_messages = [[
                {"role": "system", "content": config['system_prompt']},
                {"role": "user", "content": f"Candidate 1:\n{parent1}\n\nCandidate 2:\n{parent2}"}
            ] for (parent1, parent2) in parents]
        prompt_strings = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        inputs = tokenizer(prompt_strings, return_tensors="pt", padding=True, padding_side="left").to(generator.device)
        prompt_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            start_time = time.perf_counter()
            output = generator.generate(input_ids=prompt_ids, attention_mask=attention_mask, do_sample=True, temperature=config['temperature'], max_new_tokens=config['max_new_tokens'], use_cache=True, return_dict_in_generate=True)
            time_spent_on_LLM_GEN += time.perf_counter() - start_time 
        X_strings = tokenizer.batch_decode(output.sequences[:, prompt_ids.shape[1]:], skip_special_tokens=True)
    elif config['operator'] == 'TOKEN_GENETIC':
        X_strings = []
        for parent1, parent2 in parents:
            X_strings.append(single_point_crossover(parent1, parent2))
            X_strings[-1] = token_mutation(X_strings[-1], list(config['vocabulary']), substitution_rate=config['substitution_rate'], indel_rate=config['indel_rate'])
    for idx, X_string in enumerate(X_strings):
        if config['operator'] == 'LLM':
            match = re.search(r"<candidate>(.*?)</candidate>", X_string, re.DOTALL)
            if match:
                X_string = match.group(1).strip()
            else:
                print("ERROR: candidate was not enclosed by markers")
                X_string = X_string.strip()
        elif config['operator'] == 'TOKEN_GENETIC':
            match = True
        X_tokens = tokenizer(X_string, return_tensors='pt')['input_ids'][0:1, :config['max_new_tokens']]
        # obtain observations
        start_time = time.perf_counter()
        X_reward, _ = reward_function(X_tokens)
        X_validity = X_reward.isfinite()
        X_reward[~X_validity] = reward_for_invalid_generation 
        time_spent_on_EVAL += time.perf_counter() - start_time
        print("The instantaneous rewards are ", X_reward, " with validity ", X_validity)
        # add to population and subselect fittest individuals (steady state replacement strategy combined with truncation selection for survival)
        if match:
            print(f'add {X_string} with score {X_reward.item()} to the population')
            population.append({'candidate': X_string, 'score': X_reward.item()})
            population = sorted(population, key=lambda x: x["score"], reverse=True)[:config['population_size']]
        # record candidates & their scores
        generations[step+idx:step+idx+1, 0:X_tokens.shape[1]] = X_tokens
        rewards[step+idx:step+idx+1] = X_reward
        n_rewards_obs[step+idx:step+idx+1] += 1
        validities[step+idx:step+idx+1] = X_validity.type(torch.float)
        # resample best solution in case of stochastic rewards to ensure that we do not overfit to statistical anomaly
        best_index = rewards.argmax().item()
        if X_reward.item() < rewards[best_index]:
            best_reward, _ = reward_function(generations[best_index].unsqueeze(0))
            best_reward[~best_reward.isfinite()] = reward_for_invalid_generation
            rewards[best_index] = (rewards[best_index]*n_rewards_obs[best_index]+best_reward) / (n_rewards_obs[best_index]+1)
            n_rewards_obs[best_index] += 1

    # update metrics
    simple_reward[step//batch_size] = rewards[best_index]
    cumulative_reward[step//batch_size] = cumulative_reward[max(0, step//batch_size-1)] + torch.sum(rewards[step:step+batch_size])
    # update weights and biases
    wandb.run.summary["best_generated"] = generations[best_index]
    wandb.run.summary["best_generated_str"] = tokenizer.decode(generations[best_index, :], skip_special_tokens=True)
    wandb.run.summary["time_LLM/step"] = time_spent_on_LLM_GEN/(step+batch_size)
    wandb.run.summary["time_EVAL/step"] = time_spent_on_EVAL/(step+batch_size)
    wandb.log({"inst_reward": torch.mean(rewards[step:step+batch_size]).item(), # instantaneous reward
            "simple_reward": simple_reward[step//batch_size].item(), 
            "cumulative_reward": cumulative_reward[step//batch_size].item(),
            "validities": torch.mean((validities[step:step+batch_size] == True).to(torch.float)).item()}, 
            step=step+1)
print(f"DONE")

# Store results
torch.save({"config": config,
            "simple_reward": simple_reward, 
            "simple_reward_top10": simple_reward_top10,
            "cum_reward": cumulative_reward,
            "time_LLM_GEN": time_spent_on_LLM_GEN/config['num_samples'],
            "time_EVAL": time_spent_on_EVAL/config['num_samples'],
            "generations":generations,
            "rewards": rewards,
            "n_rewards_obs": n_rewards_obs,
            "validities": validities}, f"{config['results_dir']}/{config['seed']}-{wandb.run.name}-metrics.pt")  
wandb.finish()