# MS-Attn-Simulation
Simulation code for paper "Training Dynamics of Multi-Head Softmax Attention for In-Context Learning: Emergence, Convergence, and Optimality". Arxiv: [paperlink](https://arxiv.org/abs/2402.19442)

# Installation
requirements.tex freezes the packages used in this project. 

# Simulation
Run training.py after modifying the line 
```python
wandb.init(project="<your project name>", entity="<your entity name>", config=hparams_dict)
```