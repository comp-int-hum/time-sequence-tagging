
import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

class RNNTagger(nn.Module):

    def __init__(
            self,
            lstm_input_size,
            task_num,
            output_layers = 1,
            lstm_num_layers = 1,
            lstm_hidden_size = 512,
            mlp_layer_sizes = [256, 128],
            dropout = 0.6
    ):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        # input_size must be the same size as the bert embedding
        super(RNNTagger, self).__init__()

        self.lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
            bidirectional = False,
            dropout=dropout
        )
        
        self.classifier_heads = nn.ModuleList()
        for num in range(task_num):
            head = nn.Sequential()
            prev_size = lstm_hidden_size # * 2
            for layer_num, size in enumerate(mlp_layer_sizes):
                head.add_module(f"Task {num} Linear Layer {layer_num}", nn.Linear(in_features = prev_size, out_features = size))
                head.add_module(f"Task {num} Relu {layer_num}", nn.ReLU())
                prev_size = size
            head.add_module(f"Task {num} Output layer", nn.Linear(in_features = prev_size, out_features = 1))
            self.classifier_heads.append(head)

        self.dropout = nn.Dropout(dropout)
        

    def forward(self, input, teacher_forcing = None, teacher_ratio = 0.0, temperature = 1.0):

        if teacher_forcing is not None or teacher_ratio > 0.0:
            logger.warning(f"Teacher forcing does not apply to vanilla RNN tagger. Teacher ratio of {teacher_ratio} will be ignored.")
        
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input)
        lstm_out = self.dropout(lstm_out)
        logits = []
        for head in self.classifier_heads:
            out = head(lstm_out)  # shape: (batch, seq_len, 1)
            if temperature != 1.0:
                out = out / temperature
            logits.append(out)


        logits = torch.stack(logits, dim=2)
        logits = logits.squeeze(-1)

        return logits # (batch_size, seq_len, num_heads)