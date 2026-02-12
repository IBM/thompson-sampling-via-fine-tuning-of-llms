from typing import List, Tuple, Any
import torch, argparse, wandb, yaml, os, torch._dynamo, numpy
from datasets import load_dataset
from transformers import AutoTokenizer
os.environ["WANDB__SERVICE_WAIT"] = "300" # increases wandb allowed startup time
import torch._dynamo
torch._dynamo.config.suppress_errors = True # some models cannot be compiled throughout, so we fall back to eager execution in case compilation is not possible
torch.compile = lambda f=None, *args, **kwargs: f if f is not None else (lambda f2: f2) # disables compilation

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "1.0"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "0.0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_config(description: str, notes:str="", tags:List[str]=[]) -> dict:
    """
        Sets up weights and biases for monitoring and handling of configuration files.
        To that end, an argument parser is employed and any passed YAML configuration
        files are used to overwrite the default weights and biases configuration.

        Args:
            description (str): The ArgParse description that is displayed in the programs `help` page.
            notes (str, optional): The notes to display in weights and biases.
            tags (List[str], optional): The tags to associate with the run in weights and biases.

        Returns:
            out (dict): The run configuration identical
    """
    wandb.init(project="tosfit")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file. Overwrites the default wandb configuration file.", required=False)
    args, _ = parser.parse_known_args() # just parses the config yaml file    
    if args.config is not None: # in case a config yaml file was passed, overwrites wandb config
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        wandb.config.update(config_dict, allow_val_change=True)
    for key, value in wandb.config.items(): # enables the direct overwriting of config entries through argument parsing
        if isinstance(value, list):  # Handle lists (e.g., tags)
            parser.add_argument(f"--{key}", nargs="+", type=str, help=f"Override {key} (default: {value})")
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", type=str2bool, help=f"Override {key} (default: {value})")
        else:
            arg_type = type(value) if isinstance(value, (int, float, bool)) else str
            parser.add_argument(f"--{key}", type=arg_type, help=f"Override {key} (default: {value})")
    args = parser.parse_args()
    for key in wandb.config.keys():
        arg_value = getattr(args, key, None)
        if arg_value is not None:
            wandb.config.update({key: arg_value}, allow_val_change=True)
    wandb.run.notes = wandb.config.get('notes', description + "\n" + notes)
    wandb.run.tags = tuple(wandb.config.get("tags", tags))
    return dict(wandb.config)
        
def init_torch(seed:int) -> None:
    """
    Initializes PyTorch and ensures reproducibility.

    Args:
        seed (int): The number that seeds all randomness.
    """
    # Initializes torch
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = 64 # gives more cache to torch.compile
    # Ensures reproducibility by seeding runs
    seed_randomness(seed)
    # Logs available devices
    print(f"Runs on {torch.cuda.device_count()} GPUs of type {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}" if torch.cuda.is_available() else "Runs on CPU")

def seed_randomness(seed:int) -> None:
    """
    Ensures reproducibility by seeding all randomness.

    Args:
        seed (int): The number that seeds all randomness.
    """
    # Ensures reproducibility by seeding runs
    numpy.random.seed(numpy.abs(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_gpu(id:int) -> torch.device:
    """
        Returns the `id`-th GPU if available, otherwise falls back to GPU with largest available id or the CPU if no GPU is available.

        Args:
            id (int): The `id` of the desired GPU. Must be a natural number (starts at 1).
        
        Returns:
            out (torch.device): A pointer to a torch device.
    """
    assert id>0, f"device indexing starts at 1 but id = {id}"
    num_gpus = torch.cuda.device_count()
    feasible_id = min(id, num_gpus)
    return torch.device(f"cuda:{feasible_id-1}") if feasible_id != 0 else torch.device("cpu")

def process_prompt(seed:int, tokenizer:AutoTokenizer, prompt:str, system_prompt:str=None) -> Tuple[str, str, Any]:
    """
        Processes the prompt and system prompt to a unified prompt which can be fed to a language model. Note that
        prompt can also describe a family (dataset) of prompts from which one specific prompt is randomly sampled.
        In case the prompt is associated with a ground-truth answer, that answer is also returned.

        Args:
            seed (int): The seed for sampling or modulo selection in prompt datasets.
            tokenizer (AutoTokenizer): The tokenizer that will be used to process the prompt. Here its chat template is used to combine the `prompt` with the `system_prompt`.
            prompt (str): The prompt, or family of prompts from which we sample (supports AIME2024).
            system_prompt (str, optional): The system prompt to be used. If set to None, the prompt is returned as is. Defaults to None.
            
        Returns:
            out (Tuple[str, str, Any]): The unified prompt to be fed to the language model, the user prompt (either equals `promp` or is a sample from a family of prompts), and possibly a desired ground-truth answer (does not have to be text).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    answer = None
    # handles `prompts` that specify a whole family of prompts from which we sample
    match prompt:
        case "AIME2024": # American Invitational Mathematics Examination (AIME) 2024
            ds = load_dataset("HuggingFaceH4/aime_2024", split="train") # only has train split with 30 entries
            ds_problem = ds[torch.randint(ds.num_rows, (), generator=g).item()]
            prompt = ds_problem['problem']
            solution = ds_problem['solution']
            answer = ds_problem['answer']
        case "AIME2024%": # American Invitational Mathematics Examination (AIME) 2024
            ds = load_dataset("HuggingFaceH4/aime_2024", split="train") # only has train split with 30 entries
            ds_problem = ds[seed%30]
            prompt = ds_problem['problem']
            solution = ds_problem['solution']
            answer = ds_problem['answer']
        case "MATH500%":
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test") # only has test split with 500 entries
            ds_problem = ds[seed%500]
            prompt = ds_problem['problem']
            solution = ds_problem['solution']
            answer = ds_problem['answer']
    if system_prompt is not None:
        data = {
                "system": system_prompt,
                "query": prompt,
            }
        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
        ]
        total_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        total_prompt = prompt[:]
    return total_prompt, prompt, answer