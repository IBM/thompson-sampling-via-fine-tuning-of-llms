# ToSFiT: Thompson Sampling via Fine-Tuning of LLMs

### Nicolas Menet, Aleksandar Terzić, Michael Hersche, Andreas Krause, and Abbas Rahimi

This is the official repository of [Thompson Sampling via Fine Tuning of LLMs](https://arxiv.org/abs/2510.13328), published as a conference paper at ICLR 2026. To repeat the experiments from the paper, run the `scripts` while passing the experiment settings (`yaml` files) as `--config`. To use the codebase for your own projects, the contents of the `src` directory should suffice.

Paper: https://arxiv.org/abs/2510.13328

## 🧠 Core Idea

Bayesian optimization in exponentially large unstructured discrete spaces is intractable since the computational cost of maximizing acquisition functions scales with the domain cardinality. We propose a scalable alternative based on Thompson sampling that eliminates the need for acquisition function maximization by directly parameterizing the probability that a candidate yields the maximum reward. Our approach, Thompson Sampling via Fine-Tuning (ToSFiT), leverages the prior knowledge embedded in prompt-conditioned large language models, and incrementally adapts them toward the posterior. ToSFiT admits strong theoretical guarantees: given a sufficiently large compute budget it performs as well as widely employed classical algorithms for Bayesian optimization such as UCB or standard Thompson sampling, while being exponentially cheaper to run in practice.

## 📊 Experimental Results

Empirically, we validate our method on three diverse tasks: FAQ response refinement, thermally stable protein search, and quantum circuit design. Within a collection of methods covering Bayesian optimization, reinforcement learning, and evolutionary search, ToSFiT exhibits both state-of-the-art sample efficiency and computational efficiency.

## 📚 Citation
If you use the work released here for your research, please cite our paper:


```
@inproceedings{
menet2026thompsonsamplingfinetuningllms,
title={Thompson Sampling via Fine-Tuning of {LLM}s},
author={Nicolas Menet and Aleksandar Terzić and Michael Hersche and Andreas Krause and Abbas Rahimi},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=u6fc354gfS}
}
```

## 🕵️ Guide to Reproduce our Results

### Hardware and OS
All experiments were conducted on compute nodes with an AMD EPYC 7763 64 Core CPU, 2TB RAM, and an NVIDIA A100-SXM4 GPU (80GB), running on Red Hat Enterprise Linux 9.4 with CUDA 12.4.

### MICROMAMBA
We use `micromamba` to handle package dependencies. To create/install and activate the `tosfit` environemnt, run

> micromamba create -n tosfit -f environment.yaml

> micromamba activate tosfit

### Configuration files
In this project, we use `yaml` configuration files to describe the different experimental settings. Note however, that every entry of the config file can be overwritten by passing a key-value pair through optional arguments to the script.

### Bayesian Optimization Baselines
To compare against `Unguided Generation` and `Post-Generation TS`, use the following command:

> for i in {0..24}; do python -m scripts.run_filtering --config=experiments/baselines/protein.yaml --seed=$i; done

Since sampling from the LLMs and evaluating the proposals is rather expensive, we recommend running different Bayesian optimization algorithms or different hyperparameters (including distinct kernels feature maps) after the first seeds of full runs have finished using the following command:

> for file in experiments/baselines/results/protein/*gen.pt; do filename=$(basename "$file" -gen.pt); python -m scripts.run_filtering --config=experiments/baselines/protein.yaml --storage_id=$filename --already_generated=True --already_evaluated=True --SETTINGS_THAT_CHANGE; done

### Evolutionary Search Baselines
The two evolutionary search baselines `Evolutionary Search (LLM)` and `Evolutionary Search (Character)` can be called by using the following command:
> for i in {0..24}; do python -m scripts.run_es --config=experiments/baselines/es_llm.yaml --seed=$i; done

### Fully In-Context Bayesian Optimization Baseline
The fully in-context Bayesian optimization baseline `FIBO` can be called by using the following command:
> for i in {0..24}; do python -m scripts.run_fibo --config=experiments/baselines/fibo.yaml --seed=$i; done

### ToSFiT
To run `ToSFiT`, use the following command:
> for i in {0..24}; do python -m scripts.run_tosfit --config=experiments/tosfit/protein.yaml --seed=$i; done

### Actor Critic and Soft Actor Critic
Note that the actor critic baselines `actor critic` and `soft actor critic` can be recovered by running `ToSFiT` but setting the `exploration_bonus` to 0 and dialing up the entropy regularization coefficient `alpha` if needed.

### Collection and Visualization of Results
The results of the experiments are aggregated in their corresponding `results` directory. To compare different settings, we provide the notebook `visualizations/reward_comparison.ipynb`.