import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from functools import partial
import wandb, torch, copy
from src.utils import setup, misc
from scripts import helper
from src import language_model, gaussian_process, tosfit, lite
from src.benchmarks import reward_functions

config = setup.parse_config(description="Run Bayesian optimization using Thompson sampling through fine-tuning (tosfit).")
setup.init_torch(config['seed'])

# Configure the prompt
tokenizer = language_model.get_tokenizer(config['tokenizer'])
total_prompt, prompt, answer = setup.process_prompt(config['seed'], tokenizer, config['prompt'], config['system_prompt'])
wandb.config['prompt_sample'] = config['prompt_sample'] = prompt

# extract the feature map for kernel embeddings
feature_embedding_model, embedding_aggregation, kernel_feature_transformation, n_features = helper.feature_map(config)

# Configure the reward model
reward_model = gaussian_process.LinearGaussianProcess(n_features=n_features, 
                                                      nar=config['nar'],
                                                      device=setup.get_gpu(3), 
                                                      dtype=torch.float64)
print(f"The GP reward model uses {reward_model.memory / 1024**3} Gigabytes of memory.")

# Configure the generator and the reference generator
generator = language_model.get_model(config['generator']).eval()
#language_model.unfreeze_last_n_layers(model=generator, n=1)
for param in generator.parameters():
    param.requires_grad = True
reference_generator = copy.deepcopy(generator)
for param in reference_generator.parameters():
    param.requires_grad = False

print(f"Each LLM copy requires {misc.get_model_size(generator)} Gigabytes of memory to store the parameters (with type {list(generator.parameters())[0].dtype}).")
generator, reference_generator = torch.compile(generator.to(setup.get_gpu(1)), dynamic=True), torch.compile(reference_generator.to(setup.get_gpu(2)), dynamic=True)

# Define the optimizer
optimizer_class = torch.optim.SGD
optimizer_constructor = partial(optimizer_class, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'], maximize=True, fused=True)

# Define the reward function
reward_function_template = getattr(reward_functions, config['reward_function'])
reward_function = reward_function_template(tokenizer=tokenizer, 
                                           max_new_tokens=config['max_new_tokens'], 
                                           prompt=prompt,
                                           answer=answer, 
                                           device=setup.get_gpu(4),
                                           prm=config.get('prm', None),
                                           llm=config.get('llm', None),
                                           judge_prompt=config.get('judge_prompt', None))

# Run tosfit
simple_reward, cumulative_reward, simple_reward_top10, BoN, majority, w_BoN,\
BoN_top10, majority_top10, w_BoN_top10, pass_at_k, entropy, kl_divergence, loss, \
time_spent_on_LLM_per_step, time_spent_on_LLM_GEN_per_step, \
time_spent_on_GP_per_step, time_spent_on_EVAL_per_step, \
generations, rewards, n_rewards_obs, validities, answers = tosfit.tosfit(reward_function = reward_function,
                                                                     reward_for_invalid_generation = reward_functions.reward_for_invalid_generation[reward_function_template],
                                                                     answer = answer,
                                                                     reward_model = reward_model,
                                                                     feature_embedding_model = feature_embedding_model,
                                                                     embedding_aggregation = embedding_aggregation,
                                                                     kernel_feature_transformation = kernel_feature_transformation,
                                                                     inverse_pom_activation_exp=getattr(lite, config['inverse_pom_activation_exp']),
                                                                     total_prompt = total_prompt,
                                                                     tokenizer = tokenizer,
                                                                     generator = generator,
                                                                     reference_generator = reference_generator,
                                                                     temperature = config['temperature'],
                                                                     max_new_tokens = config['max_new_tokens'],
                                                                     optimizer_constructor = optimizer_constructor,
                                                                     num_samples = config['num_samples'],
                                                                     bo_batch_size = config['bo_batch_size'],
                                                                     fine_tune_steps_per_bo_step = config['fine_tune_steps/bo_step'],
                                                                     batch_size = config['batch_size'],
                                                                     mini_batch_size = config['mini_batch_size'],
                                                                     alpha = config['alpha'],
                                                                     observe_invalid_generations = config['observe_invalid_generations'],
                                                                     n_marginal_likelihood_warmup_steps = config['n_marginal_likelihood_warmup_steps'],
                                                                     ongoing_marginal_likelihood_maximization = config['ongoing_marginal_likelihood_maximization'],
                                                                     exploration_bonus = config['exploration_bonus'])

# Store results
torch.save({"config": config,
            "simple_reward": simple_reward, 
            "simple_reward_top10": simple_reward_top10,
            "cum_reward": cumulative_reward,
            "BoN": BoN,
            "BoN_top10": BoN_top10,
            "majority": majority,
            "majority_top10": majority_top10,
            "w_BoN": w_BoN,
            "w_BoN_top10": w_BoN_top10,
            "pass_at_k": pass_at_k,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "loss": loss,
            "time_LLM": time_spent_on_LLM_per_step,
            "time_LLM_GEN": time_spent_on_LLM_GEN_per_step,
            "time_GP": time_spent_on_GP_per_step,
            "time_EVAL": time_spent_on_EVAL_per_step,
            "generations":generations,
            "answers": answers,
            "rewards": rewards,
            "n_rewards_obs": n_rewards_obs,
            "validities": validities}, f"{config['results_dir']}/{config['seed']}-{wandb.run.name}-metrics.pt")  
wandb.finish()