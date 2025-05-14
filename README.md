# FlowerTune LLM Benchmark

This directory conducts federated instruction tuning with different pre-trained LLMs on four challenges defined in the [FlowerTune LLM Leaderboard](https://flower.ai/benchmarks/llm-leaderboard): general NLP, finance, medical and code.
The experiments in paper "FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models" are conducted using this repository.

We use [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the dataset.
[Flower](https://flower.ai)'s Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Environments setup

In each project directory, the dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
cd project_name  # selected from [general_nlp, finance, medical, coding]

pip install -e .
pip install flash-attn --no-build-isolation   # Install FlashAttention-2
```

## Running federated fine-tuning

First make sure that you have got the access to your preferred model with your Hugging-Face account. You can request access directly from the Hugging-Face website.
Then, follow the instruction [here](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) to log in your account. Note you only need to complete this stage once in your development machine:

```bash
huggingface-cli login
```

Then, login your W&B account if you want to use it for experimental status logging.
To disable W&B, set `use-wandb = false` in `pyproject.toml`.

```bash
wandb login
```

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

To run a specified experiment:

```bash
# Run on Mistral-7B-v0.3 model without wandb
flwr run --run-config "model.name='mistralai/Mistral-7B-v0.3' run-name='customised_name' use-wandb=false"

# Run with FedProx
flwr run --run-config "strategy.name='fedprox'"

# Run with LoRA
flwr run --run-config "model.lora.peft-use-dora=false"
```

### Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under `[tool.flwr.app.config]` entry in `pyproject.toml`.


## Experimental model checkpoints

The model checkpoints fine-tuned in paper "FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models" can be found here: 
[General-NLP](https://huggingface.co/collections/yangao381/flowertune-general-nlp-68246cd84668df78a8ff5043), 
[Finance](https://huggingface.co/collections/yangao381/flowertune-finance-682488597020f86deaa67b10), 
[Medical](https://huggingface.co/collections/yangao381/flowertune-medical-68248900e2dfe912f34a5b81), 
[Code](https://huggingface.co/collections/yangao381/flowertune-code-682489a2bdadc846697600dc).


## Running the evaluation

To evaluate the fine-tuned LLMs, please follow the instructions in the [FlowerTune Evaluation GitHub](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation).
