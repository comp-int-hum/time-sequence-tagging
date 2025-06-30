from torch.nn import Linear, Identity, Module
import torch.nn as nn
import torch
import math
import random

class HRNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, layer_names = [], dropout = 0., device = "cpu"):
        super().__init__()
        
        self.hidden_dim = hidden_size
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        
        self.dropout = nn.Dropout(p = dropout)
        
        self.layer_names = layer_names
        
        if layer_names:
            assert len(layer_names) == (num_layers - 1)

        self.hrnn_cell = HRNNCell(input_size, hidden_size, num_layers, dropout = dropout, device = device)
        
    def forward(self, x, teacher_forcing = None, teacher_ratio = 0.0, temperature = 1.0):
        """
        x: Tensor(batch_size, sequence_len, input_size)
        teacher_forcing: Tensor(batch_size, sequence_len, num_layers)
        """
        hidden_states = self.initialize_hidden_states(x.shape[0])
        boundary_preds = []
        for t in range(x.shape[1]):
            x_t = self.dropout(x[:, t, :])
            if random.random() <= teacher_ratio:
                hidden_states, boundary_pred = self.hrnn_cell(x_t, hidden_states, teacher_forcing = teacher_forcing[:, t, :], temperature = temperature)
            else:
                hidden_states, boundary_pred = self.hrnn_cell(x_t, hidden_states, teacher_forcing = None, temperature = temperature)
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
        self.dropout = nn.Dropout(p = dropout)
        # self.temperature = torch.tensor(temperature, device = device, dtype = torch.float32)
        
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
        
    def forward(self, x, hidden_states, teacher_forcing = None, temperature = 1.0):
        """
        x: (batch_size, input_size)
        hidden_states: List[Tensor(batch_size, hidden_size)]
        teacher_forcing: Tensor(batch_size, num_layers)
        """
        
        # print(f"Type of hidden states: {type(hidden_states)}")
        
        temperature = torch.tensor(temperature, device = self.device, dtype = x.dtype)
        batch_size = x.shape[0]
        # updated_hidden_states = [torch.zeros_like(hidden_states[0]) for _ in range(self.num_layers)]
        hs_mixture_weights = [[] for _ in range(self.num_layers)]
        candidate_hidden_states = [[] for _ in range(self.num_layers)]
        
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
            
            # Get predicted transition probability
            transition_pred = torch.sigmoid(self.transition_mlps[l](input_to_cell)) if l != self.num_layers-1 else torch.zeros((batch_size, 1), device = self.device)
            
            # Set transition probability to model prediction or teacher label
            transition_prob = teacher_forcing[:, l].unsqueeze(dim = 1) if (teacher_forcing is not None and l < teacher_forcing.shape[1]) else transition_pred
            
            # Update hidden states h <= l
            for h in range(l + 1):
                cum_pred = torch.prod(torch.stack(cum_transition_probs[h : l]), dim = 0) if cum_transition_probs[h:l] else torch.ones((batch_size, 1), device = self.device)
                # updated_hidden_states[h] +=  cum_pred * (1 - transition_prob) * cell_outputs[l]
                hs_mixture_weights[h].append(cum_pred * (1 - transition_prob))
                candidate_hidden_states[h].append(cell_outputs[l])
            
            # update for next layer
            cum_transition_probs.append(transition_prob)
            cum_transition_preds.append(transition_pred)
            
        hs_mixture_probs = [nn.functional.softmax(torch.stack(hs_mixture_weights[lnum]) / temperature,  dim = 0) 
                            for lnum in range(self.num_layers)] # List[Tensor(layers, batch_size, 1)]
            
        updated_hidden_states = [torch.sum(hs_mixture_probs[lnum] * torch.stack(candidate_hidden_states[lnum]), dim = 0) for lnum in range(self.num_layers)]

        return updated_hidden_states, torch.cat(cum_transition_preds[:-1], dim = 1)
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout = 0.6, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p = dropout))
            
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)