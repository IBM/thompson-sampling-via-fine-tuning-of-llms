import wandb, torch, math, time, re
from src.utils import setup, misc
from src import language_model
from src.benchmarks import reward_functions

config = setup.parse_config(description="Run Bayesian optimization using FIBO.")
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

# statistics
time_spent_on_LLM_GEN = 0
time_spent_on_EVAL = 0

# run FIBO
prompt_reward, _ = reward_function(tokenizer(config['prompt'], return_tensors="pt")['input_ids']) # prompt contains example to illustrate behaviour.
messages = [
    {"role": "system", "content": config['system_prompt']},
    {"role": "assistant", "content": "**Candidate solution:**\n<candidate>" + config['prompt'] + "</candidate>"},
    {"role": "user", "content": f"**Reward associated with the above candidate solution:\n** {prompt_reward.item()}"}
]
history = [(messages[-2], messages[-1], prompt_reward.item())] # assistant, user, reward
#past_key_values = None # cache
for step in range(0, config['num_samples'], batch_size):
    print(f"FIBO step {step}/{config['num_samples']}")
    start_time = time.perf_counter()
    with torch.no_grad():
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs['input_ids'].to(generator.device).repeat(batch_size, 1)
        attention_mask = inputs['attention_mask'].to(generator.device).repeat(batch_size, 1)
        output = generator.generate(input_ids=prompt_ids, attention_mask=attention_mask, do_sample=True, temperature=config['temperature'], max_new_tokens=config['max_new_tokens'], use_cache=True, return_dict_in_generate=True)#, past_key_values=past_key_values)
        X_strings = tokenizer.batch_decode(output.sequences[:, prompt_ids.shape[1]:], skip_special_tokens=True) # responses without prompt
        print('X_STRINGS:\n', X_strings)
        #print('prompt_ids.shape', prompt_ids.shape)
        #print('full_output', tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0])
    time_spent_on_LLM_GEN += time.perf_counter() - start_time 
    for idx, X_string in enumerate(X_strings):
        match = re.search(r"<candidate>(.*?)</candidate>", X_string, re.DOTALL)
        if match:
            X_string = match.group(1).strip()
        else:
            print("ERROR: candidate was not enclosed by markers")
            X_string = X_string.strip()
        X_tokens = tokenizer(X_string, return_tensors='pt')['input_ids'][0:1, :config['max_new_tokens']]
        # obtain observations
        start_time = time.perf_counter()
        X_reward, _ = reward_function(X_tokens)
        X_validity = X_reward.isfinite()
        X_reward[~X_validity] = reward_for_invalid_generation 
        time_spent_on_EVAL += time.perf_counter() - start_time
        print("The instantaneous rewards are ", X_reward, " with validity ", X_validity)
        # add 
        if match:
            messages += [
                {"role": "assistant", "content": "**Candidate solution:**\n<candidate>" + X_string + "</candidate>"},
                {"role": "user", "content": f"**Reward associated with the above candidate solution:\n** {X_reward.item()}"}
            ]
            history += [(messages[-2], messages[-1], X_reward.item())] # assistant, user, reward
        if config.get('memory_size', None):
            messages = messages[0:1] + messages[1:][-2*config['memory_size']:] # system prompt + last few message pairs if short term memory is enabled to avoid OOM errors.
        if config.get('top_o', None):
            history = list({ass['content']: (ass, usr, rwrd) for ass, usr, rwrd in history}.values()) # history only contains unique responses to avoid collapse to a single observation
            history = sorted(history, key=lambda x: x[2])[-config['top_o']:] # only retain top_o entries in history and sort them increasingly to nudge the LLM to further improve the solution
            messages = messages[0:1] + [msg for (ass, usr, rwrd) in history for msg in (ass, usr)] # system prompt + best message pairs if top_o is enabled to avoid OOM errors.
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
            "cum_reward": cumulative_reward,
            "time_LLM_GEN": time_spent_on_LLM_GEN/config['num_samples'],
            "time_EVAL": time_spent_on_EVAL/config['num_samples'],
            "generations":generations,
            "rewards": rewards,
            "n_rewards_obs": n_rewards_obs,
            "validities": validities}, f"{config['results_dir']}/{config['seed']}-{wandb.run.name}-metrics.pt")  
wandb.finish()
