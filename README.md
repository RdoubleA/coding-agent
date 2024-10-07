# coding-agent
Python coding agent trained with torchtune

## Setup
Install torchtune
```
pip install torchtune
```

## Launch training
```
tune run --nproc_per_node 8 sft --config config/sft_llama3_2_1B.yaml
```

You can see an example training run here: https://wandb.ai/rafi-personal/coding-agent/workspace

## Launch offline generation/eval
```
tune run offline_eval --config config/offline_eval_llama3_2_1B.yaml
```
