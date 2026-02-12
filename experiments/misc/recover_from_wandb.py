import wandb
import torch

def recover_wandb_runs(entity, project, constraints:dict):
    """
    Recovers all tosfit runs from a Weights & Biases project that match the specified criteria.
    
    Args:
        entity (str): The W&B entity (username or team name).
        project (str): The W&B project name.
        constraints (dict): Only retrieve runs whose config matches the constraints are retrieved.
    """
    # Authenticate with W&B
    wandb.login()
    api = wandb.Api()
    # Get all runs in the project
    runs = api.runs(path=f"{entity}/{project}", filters=constraints, include_sweeps=False)

    for run in runs:
        # Filter runs based on criteria
        print("recovering run")
        storage_path = f"experiments/tosfit/results/{run.config['reward_function']}/{run.name}"
        run_history = run.scan_history(keys=["highest_reward", "cumulative_reward", "generator_entropy", "kl_divergence"])
        # Iterate through the generator and print each value
        torch.save({"best_rewards": torch.tensor([row["highest_reward"] for row in run_history]), 
            "cum_rewards": torch.tensor([row["cumulative_reward"] for row in run_history]),
            "entropy": torch.tensor([row["generator_entropy"] for row in run_history]),
            "kl_divergence": torch.tensor([row["kl_divergence"] for row in run_history]),
            "config": dict(run.config)}, storage_path + "-tosfit-rewards.pt")  

# Example usage
if __name__ == "__main__":
    entity = "" 
    project = "tosfit" 
    constraints = {'state': 'finished',
                   'config.learning_rate': 0.00002, 
                   'config.kernel_feature_map': 'mean_token_embedding',}
    
    recover_wandb_runs(entity, project, constraints)