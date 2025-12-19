import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SimpleTagger(nn.Module):
    def __init__(
            self,
            input_size,
            task_num,
            task_size=1,
            mlp_layer_sizes=[256, 128],
            dropout=0.6
    ):
        """
        Args:
            input_size: Dimension of each token embedding (e.g., from BERT)
            task_num: Number of tagging heads (e.g., paragraphs, chapters)
            task_size: Output size per head (1 for binary classification)
            mlp_layer_sizes: List of hidden sizes for each MLP layer
            dropout: Dropout probability
        """
        super(SimpleTagger, self).__init__()

        self.classifier_heads = nn.ModuleList()
        for num in range(task_num):
            head = nn.Sequential()
            prev_size = input_size
            for layer_num, size in enumerate(mlp_layer_sizes):
                head.add_module(f"Task {num} Linear Layer {layer_num}", nn.Linear(prev_size, size))
                head.add_module(f"Task {num} ReLU {layer_num}", nn.ReLU())
                prev_size = size
            head.add_module(f"Task {num} Output Layer", nn.Linear(prev_size, task_size))
            self.classifier_heads.append(head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, teacher_forcing=None, teacher_ratio=0.0, temperature=1.0):
        """
        Args:
            input: Tensor of shape (batch_size, seq_len, input_size)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, num_heads)
        """
        if teacher_forcing is not None or teacher_ratio > 0.0:
            logger.warning(f"Teacher forcing does not apply to non-recurrent tagger. Teacher ratio of {teacher_ratio} will be ignored.")

        batch_size, seq_len, input_dim = input.shape
        input = self.dropout(input)

        logits = []
        for head in self.classifier_heads:
            # Flatten the sequence dimension into the batch
            flattened_input = input.view(-1, input_dim)  # shape: (batch * seq_len, input_dim)
            out = head(flattened_input)                  # shape: (batch * seq_len, task_size)
            
            if temperature != 1.0:
                out = out / temperature

            # Reshape back to (batch, seq_len, task_size)
            out = out.view(batch_size, seq_len, -1)
            logits.append(out)

        # Stack across num_heads → (batch, seq_len, num_heads, task_size)
        logits = torch.stack(logits, dim=2)

        # Squeeze final dim if task_size == 1
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)  # (batch, seq_len, num_heads)

        return logits