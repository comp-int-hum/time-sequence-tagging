
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay


class SequenceTagger(nn.Module):

    def __init__(
            self,
            task_sizes,
            lstm_input_size,
            output_layers = 1,
            lstm_num_layers = 1,
            lstm_hidden_size = 512,
            mlp_layer_sizes = [256, 128],
            dropout = 0.4
    ):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        # input_size must be the same size as the bert embedding
        super(SequenceTagger, self).__init__()

        self.lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
            bidirectional = True,
            dropout=dropout
        )
        
        self.classifier_heads = nn.ModuleList()
        for task_num, task_size in enumerate(task_sizes):
            head = nn.Sequential()
            prev_size = lstm_hidden_size * 2
            for layer_num, size in enumerate(mlp_layer_sizes):
                head.add_module(f"Task {task_num} Linear Layer {layer_num}", nn.Linear(in_features = prev_size, out_features = size))
                head.add_module(f"Task {task_num} Relu {layer_num}", nn.ReLU())
                prev_size = size
            head.add_module(f"Task {task_num} Output layer", nn.Linear(in_features = prev_size, out_features = task_size))
            self.classifier_heads.append(head)
        self.dropout = nn.Dropout(dropout)
        
        # Define loss function
        #self.loss_fn = nn.CrossEntropyLoss()
        

    def forward(self, sentence_embeds, device = "cpu"):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(sentence_embeds)
        lstm_out = self.dropout(lstm_out)
        outputs = torch.stack([head(lstm_out) for head in self.classifier_heads], 1)
        return outputs
        
        # if labels != None:
        #     flattened_outputs, flattened_labels = process_task_outputs_and_labels(outputs, labels, self.classes, to_cpu = False)
        #     #print(flattened_outputs.shape)
        #     #print(flattened_labels.shape)
        #     #print(outputs.shape)
        #     loss, tasks_preds = self.compute_loss_and_prediction(flattened_outputs, flattened_labels, return_predictions)
            
        #     result = (loss, tasks_preds) if return_predictions else (loss,)
            
        #     data = (flattened_outputs, flattened_labels) if flatten else (outputs, labels)
        #     return result + (data,)
     
        # return outputs # (N, L, num_classes)
    
    def compute_loss_and_prediction(self, outputs, labels, return_pred = False):
        loss = 0
        tasks_predictions = []
        
        for output, label, weight in zip(outputs, labels, self.class_weights):
            if weight:
                loss += weight * self.loss_fn(output, label)
                if return_pred:
                    pred = torch.max(output, dim = 1) if output.numel() > 0 else (torch.tensor([]), torch.tensor([]))
                    predicted_class, prediction_score = pred[0].cpu().tolist(), pred[1].cpu().tolist()
                    tasks_predictions.append((predicted_class, prediction_score))
                    
        return loss, tasks_predictions
    

def reshape_output_and_labels(output, label, num_classes=2, to_cpu = False):
    output = output.view(-1, num_classes) if num_classes else torch.empty(0, dtype=torch.long)
    label = label.view(-1) if num_classes else torch.empty(0, dtype=torch.long)
    
    if num_classes == 2:
        label = (label > 0).long()
    
    return output, label.cpu().tolist() if to_cpu else label

def process_task_outputs_and_labels(task_outputs, task_labels, task_classes, to_cpu=False):
    return zip(*[reshape_output_and_labels(output, label, classes, to_cpu)
                 for output, label, classes in zip(task_outputs, task_labels, task_classes)])