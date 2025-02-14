from torch.nn import Linear, Identity, Module
import torch.nn as nn
import torch
import math
import random

class GenericHRNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0., device = "cpu"):
        super().__init__()
        
        self.hidden_dim = hidden_size
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.hrnn_cell = HRNNCell(input_size, hidden_size, num_layers, dropout = dropout, device = device)
        
    def forward(self, x, teacher_forcing = None, teacher_ratio = 0.0):
        """
        x: Tensor(batch_size, sequence_len, input_size)
        teacher_forcing: Tensor(batch_size, sequence_len, num_layers)
        """
        hidden_states = self.initialize_hidden_states(x.shape[0])
        boundary_preds = []
        for t in range(x.shape[1]):
            if random.random() <= teacher_ratio:
                hidden_states, boundary_pred = self.hrnn_cell(x[:, t, :], hidden_states, teacher_forcing = teacher_forcing[:, t, :])
            else:
                hidden_states, boundary_pred = self.hrnn_cell(x[:, t, :], hidden_states)
            # print(f"Boundary pred: {boundary_pred.shape}")
            # print(boundary_pred)
            boundary_preds.append(boundary_pred)
            
        result = torch.stack(boundary_preds, dim = 1)
        # print(f"Result shape: {result.shape}")
        return result
    
    def initialize_hidden_states(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.device) for _ in range(self.num_layers)]
    
    
class HRNNCell(Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0., device = "cpu", cell_type = "gru"):
        super().__init__()
        
        self.hidden_dim = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        if cell_type not in ["gru", "lstm"]:
            raise 

        self.cells = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size).to(device)
            for i in range(num_layers)
        ])

        self.transition_mlps = nn.ModuleList([
            MLP(input_size = hidden_size, 
                hidden_size = hidden_size, 
                output_size = 1,
                num_layers=2,
                activation_fn=nn.ReLU()).to(device)
            for _ in range(num_layers-1)
        ])
        
    def forward(self, x, hidden_states, teacher_forcing = None):
        """
        x: (batch_size, input_size)
        hidden_states: (batch_size, hidden_size)
        teacher_forcing: Tensor(batch_size, num_layers)
        """
        
        batch_size = x.shape[0]
        updated_hidden_states = [torch.zeros_like(hidden_states[0]) for _ in range(self.num_layers)]
        
        # print(f"Hidden state 0 shape: {hidden_states[0].shape}")
        # if teacher_forcing is not None:
        #     print(f"teacher forcing shape: {teacher_forcing.shape}")
        cum_transition_probs = []
        cum_transition_preds = []
        
        input_to_cell = x
        cell_outputs = []
        # iterate over original hidden layers
        for l in range(self.num_layers):
            input_to_cell = self.cells[l](input_to_cell, hidden_states[l]) # output -> (batch_size, hidden_size)
            cell_outputs.append(input_to_cell)
            transition_pred = torch.sigmoid(self.transition_mlps[l](input_to_cell)) if l != self.num_layers-1 else torch.zeros((batch_size, 1), device = self.device)
            # if isinstance(transition_pred, torch.Tensor) and transition_pred.size(-1) == 1:
            #     print(f"Transition pred: {transition_pred.shape}")
            #     transition_pred = transition_pred.squeeze(dim = -1)
            
            transition_prob = teacher_forcing[:, l].unsqueeze(dim = 1) if (teacher_forcing is not None and l < teacher_forcing.shape[1]) else transition_pred
            
            # update hidden states for layers m <= l
            for m in range(l + 1):
                cum_pred = torch.prod(torch.stack(cum_transition_probs[m : l]), dim = 0) if cum_transition_probs[m:l] else torch.ones((batch_size, 1), device = self.device)
                # print(f"Cumulative pred shape: {cum_pred.shape}")
                # if isinstance(transition_prob, torch.Tensor):
                #     print(f"Transition prob shape: {transition_prob.shape}")
                #     print("yes here")
                # else:
                #     print(f"Transition prob is zero")
                # print(f"Cell output shape: {cell_outputs[l].shape}")
                updated_hidden_states[m] +=  cum_pred * (1 - transition_prob) * cell_outputs[l]
            
            # update for next layer
            # print(f"Transition pred: {transition_pred.shape}")
            cum_transition_probs.append(transition_prob)
            cum_transition_preds.append(transition_pred)

        return updated_hidden_states, torch.cat(cum_transition_preds[:-1], dim = 1)
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)