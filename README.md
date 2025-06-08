# Sparse Mixture of Experts (MoE) for Nonlinear Regression
This project demonstrates how to use a sparsely activated Mixture of Experts model to solve nonlinear regression problems, and introduces a load balancing loss to improve the model’s generalization ability.


### Highlights
- Use multiple neural network experts to learn different sub-tasks separately.
- Dynamically select the most suitable Top-k experts via a gating network.
- Introduce load balancing loss to ensure more even usage of experts.


### Project Structure
├── moe_regression.ipynb     
└── README.md            


### Model Architecture
1. Expert Network  
Each expert is a simple 2-layer feedforward neural network used for nonlinear modeling of the input.

2. Gating Network  
The gating network calculates the activation probability of each expert based on the input and selects the Top-k experts to participate in computation.

3. SparseMoE Main Model  
Uses `torch.topk` to implement sparse selection (only activates part of the experts);  
Performs weighted summation over the outputs of the Top-k experts to generate the final result;  
Adds a load balancing loss during training to prevent biased expert assignment.


### Functionality
This code implements a sparsely activated MoE model with the following features:

- Nonlinear function fitting: learn and predict nonlinear functions;  
- Sparse activation (Top-k routing): only activate the most relevant Top-k experts for each input to improve efficiency;  
- Load balancing loss: introduce an auxiliary loss to encourage the gating network to utilize all experts fairly;  
- Training process tracking: periodically print main loss and load balancing loss during training for debugging;  
- Visualization of fitting performance: plot the model's predictions on the test set and compare with real data.


### Relation to DeepSeek
This project demonstrates and reproduces the key mechanisms of the Mixture of Experts (MoE) architecture used in the large-scale generative model DeepSeek, including sparse expert routing and load balancing strategy.  
This code is a simplified implementation of the DeepSeek-MoE concept, making it easier to understand its core principles.

- Sparse activation mechanism: uses `torch.topk` to activate Top-k experts, only activates a small number of experts, reduces computation, improves inference efficiency  
- Load balancing strategy: constructs load balancing loss based on expert selection frequency, prevents certain experts from being unused for long periods, improves generalization ability  
- Gating network structure: single-layer linear layer + Softmax to output expert selection probabilities, functionally consistent with DeepSeek’s gating network  
- Routing training mechanism: simultaneously optimizes regression loss + load balancing loss during training, DeepSeek also introduces auxiliary loss to adjust routing
