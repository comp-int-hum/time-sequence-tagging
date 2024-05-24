import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import json
import os
from utility import make_dirs
import gzip
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import logging


class SequenceTagger(nn.Module):

    def __init__(self, input_size, num_classes, hidden_dim = 512):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        # input_size must be the same size as the bert embedding
        assert input_size == 768
        super(SequenceTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim, num_layers = 1, batch_first = True, bidirectional = True)
        self.output = nn.Linear(in_features = hidden_dim * 2, out_features = num_classes)

    def forward(self, sentence_embeds, device = "cpu"):
        self.lstm.flatten_parameters() # input is (32, SEQ_LEN, 768)
        lstm_out, _ = self.lstm(sentence_embeds) # lstm_out is (batch_size, seq_len, 2 * hidden_dim)
        outputs = self.output(lstm_out)
        return outputs # (N, L, num_classes)

class SequenceTaggerWithBahdanauAttention(nn.Module):

    def __init__(self, input_size, num_classes, hidden_dim = 512):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        super(SequenceTaggerWithBahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        attention_input_size = input_size * 2
        self.forward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = 1, batch_first = True, bidirectional = False)
        self.backward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = 1, batch_first = True, bidirectional = False)
        self.output = nn.Linear(in_features = hidden_dim * 2, out_features = num_classes)

        # Implement hidden size
        self.encoder_attention = nn.Linear(in_features = input_size, out_features = hidden_dim)
        self.decoder_attention = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, sentence_embeds, device = "cpu"):
        self.forward_lstm.flatten_parameters() # input is (32, SEQ_LEN, 768)
        self.backward_lstm.flatten_parameters()
        batch_size = sentence_embeds.size(0)

        # Forward pass
        forward_outputs = []
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1) # match to [batch_size, seq_len, 768]; weighted sum across embeds in seq
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1) # unsqueeze to [N, 1, input_size]
            # print(f"contexted input: {contextualized_input.shape}")
            # print(f"h_n: {h_n.shape}")
            # print(f"c_n: {c_n.shape}")
            output, (h_n, c_n) = self.forward_lstm(contextualized_input, (h_n, c_n))
            forward_outputs.append(output.squeeze(dim = 1)) # convert to [batch, hidden_dim]

        # Backward pass
        backward_outputs = []
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)-1, -1, -1):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1)
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1)
            output, (h_n, c_n) = self.backward_lstm(contextualized_input, (h_n, c_n))
            backward_outputs.append(output.squeeze(dim = 1))
        backward_outputs.reverse()

        # Combine forward and backward
        # forward output shape: [batch, hidden_dim]
        combined_outputs = [torch.cat((f, b), dim = 1) for f, b in zip(forward_outputs, backward_outputs)]
        preds = [self.output(result) for result in combined_outputs]
        
        return torch.stack(preds, dim = 1) # [batch_size, seq_len, output_classes]
    
    def calculate_attention_scores(self, encoder_embeddings, decoder_hidden):
        # Input sizes:
            # encoder_embeddings: [batch_size, seq_len, input_size]
            # decoder_hidden: [batch_size, decoder_dim]
        enc_proj = self.encoder_attention(encoder_embeddings) # [batch_size, seq_len, hidden_dim]
        dec_proj = self.decoder_attention(decoder_hidden).squeeze(dim = 0) # [batch_size, hidden_dim]

        dec_proj = dec_proj.unsqueeze(dim = 1).expand_as(enc_proj) # expand to [batch_size, seq_len, hidden_dim]

        tanh_result = torch.tanh(enc_proj + dec_proj) # [batch_size, seq_len, hidden_dim]
        pre_scores = self.v * tanh_result # broadcast [hidden_size] to multiply with [batch_size, seq_len, hidden_dim]
        scores = torch.sum(pre_scores, dim = 2) # [batch_size, seq_len]
        return scores
    
    def get_initial_states(self, batch_size, device = "cpu"):
        h_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        c_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        return (h_0, c_0)


def get_batch_data(datapoint):
    data = datapoint["sequence_embeds"]
    label = datapoint["labels"]
    metadata = {"id": datapoint["id"],
                "original_text" : datapoint["original_text"],
                "paragraph": datapoint["paragraph"],
                "labels": datapoint["labels"]}

    return data, label, metadata


def get_batch(filepath, batch_size=32, device="cuda"):
    data_batches, label_batches, metadata_batches = [], [], []
    batch_data, batch_label, batch_metadata = [], [], []
    total_datapoints = 0
    positive_datapoints = 0
    
    # Helper function for appending batches and padding if model_type is sequence_tagger
    def append_data(batch_data, batch_label, batch_metadata):
        batch_data = rnn_utils.pad_sequence([torch.tensor(d) for d in batch_data], batch_first=True)
        batch_label = rnn_utils.pad_sequence([torch.tensor(l) for l in batch_label], batch_first=True)
        data_batches.append(batch_data.to(device))
        label_batches.append(batch_label.to(device))
        metadata_batches.append(batch_metadata)
        
    # Open data file
    with gzip.open(filepath, 'r') as gzipfile:
        datapoints = jsonlines.Reader(gzipfile)
        for datapoint in datapoints:
            total_datapoints += 1
            data, label, metadata = get_batch_data(datapoint)
            
            # Count number of positive datapoints
            if 1 in label:
                positive_datapoints += 1
                
            # Append data
            batch_data.append(data)
            batch_label.append(label)
            batch_metadata.append(metadata)
            
            # Add batch if batch_sized has been reached
            if len(batch_data) == batch_size:
                append_data(batch_data, batch_label, batch_metadata)
                batch_data, batch_label, batch_metadata = [], [], []
        
        # Add leftover data items to a batch
        if batch_data:
            append_data(batch_data, batch_label, batch_metadata)

    print(f"NUM DATAPOINTS IN FILE {total_datapoints}")
    return (data_batches, label_batches), metadata_batches, positive_datapoints, total_datapoints

def evaluate(model, batches, metadata, classes, device):
    model.eval()
    num_correct = 0
    total = 0
    incorrect_texts = []
    
    input_data, input_labels = batches
    assert len(metadata) == len(input_data)
    assert len(metadata) == len(input_labels)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input, label, meta in zip(input_data, input_labels, metadata):
            # If empty batch, pass
            if input.size(0) == 0 or label.size(0) == 0:
                print("Empty batch")
                continue

            input = input.to(device)
            label = label.to(device)

            # Output and loss
            output = model(input, device = device)
            reshaped_output, reshaped_label = reshape_output_label(output, label, len(classes))

            
            correct, curr_total, incorrect = calculate_accuracy(output, label, meta)
            
            preds = torch.argmax(reshaped_output, dim = 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(reshaped_label.cpu().tolist())
            # Get metrics
            num_correct += correct
            total += curr_total
            
            # Add incorrect texts
            incorrect_texts.extend(incorrect)
    
    # Calculate final metrics
    accuracy = num_correct / total
    # print(f"All labels: {all_labels}")
    # print(f"All preds: {all_preds}")
    cm = confusion_matrix(y_true = all_labels, y_pred = all_preds, labels = classes)
    return accuracy, cm, incorrect_texts
  
def reshape_output_label(output, label, num_classes=2):
    output = output.view(-1, num_classes) # (N, L, num_classes) -> (N x L, num_classes)
    label = label.view(-1) # (N, L) -> (N x L)
    return output, label

# def update_error(error_list, pred, label):
#     if pred == 
def calculate_accuracy(output, labels, metadata):
    incorrect_texts = []

    # Get predictions and incorrectly predicted sequences
    predicted_classes = torch.argmax(output, dim=2)
    correct_predictions = (predicted_classes == labels)
    num_correct = correct_predictions.sum().item()
    total = torch.numel(predicted_classes)

    # Iterate over batches and gather incorrect texts
    for i, correct_item in enumerate(correct_predictions): # (N, L)
        if not correct_item.all():
            error_idxs = (~correct_item).nonzero(as_tuple=False).view(-1).tolist()
            metadata[i]["errors"] = [(labels[i, idx].item(), predicted_classes[i, idx].item(), metadata[i]["original_text"][idx]) for idx in error_idxs]
            incorrect_texts.append(metadata[i])

    return num_correct, total, incorrect_texts
    
def plot_confusion_matrix(cm, save_name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(save_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", help="name of training datapoints file")
    parser.add_argument("--test", dest="test", help="name of test datapoints file")
    parser.add_argument("--model", dest="model", help="Type of model, classifier vs sequence_tagger")
    parser.add_argument("--model_name", dest="model_name", help="Name of best model")
    parser.add_argument("--emb_dim", dest="emb_dim", type = int, help="size of sentence embedding")
    parser.add_argument("--num_epochs", dest="epochs", type = int, help="number of epochs to train")
    parser.add_argument("--result", dest="result", help="Name of result file")
    parser.add_argument("--errors", dest="errors", help="Name of result file")
    parser.add_argument("--batch", dest="batch", type = int, default=32, help="Batch size")
    parser.add_argument("--classes", default=[0, 1, 2, 3], help = "What the class labels should look like")
    parser.add_argument("--confusion", dest="cm", help="Name for confusion matrix file")
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    make_dirs(args.model_name)
    make_dirs(args.result)

    print(f"**Is training**")

    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.model_name), exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_epochs = args.epochs
    best_accuracy = 0

    num_classes = len(args.classes)
    # Get batches
    train_batches, train_metadata, train_positive, train_size = get_batch(args.train, device = device)
    test_batches, test_metadata, test_positive, test_size = get_batch(args.test, batch_size=1, device = device)


    if "sequence_tagger_with_bahdanau_attention" == args.model:
        model = SequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, num_classes = num_classes)
    else:
        model = SequenceTagger(input_size = args.emb_dim, num_classes=num_classes)
            
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, )
    with open(args.result, "w") as file:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            input_len = 0
            
            # Train
            for input, label in zip(*train_batches):
                optimizer.zero_grad()
                input = input.to(device)
                label = label.to(device)
                output = model(input, device = device)
                output, label = reshape_output_label(output, label, num_classes)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                input_len += label.size(0)
            
            print(f"Epoch: {epoch}, Loss: {running_loss / input_len}")
            file.write(f"Epoch: {epoch}, Loss: {running_loss / input_len}\n")
                
            # Eval
            print(f"Test batches: {len(test_batches)}")
            print(f"Test batches metadata: {len(test_metadata)}")
            accuracy, cm, incorrect_texts = evaluate(model, test_batches, test_metadata, args.classes, device)
            print(f"Type of incorrect_texts: {type(incorrect_texts)}")
            print(f"Accuracy: {accuracy:.2f}")
            file.write(f"Accuracy: {accuracy:.2f}\n")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), args.model_name)
        print("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))
        file.write(f"Train size: {train_size} \n")
        file.write(f"Train positive: {train_positive} \n")
        file.write(f"Test size: {test_size} \n")
        file.write(f"Test positive: {test_positive} \n")
        file.write(f"Number of incorrect texts: {len(incorrect_texts)}")
        file.write("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))

    with open(args.errors, "w") as errors:
        errors.write(json.dumps(incorrect_texts))
    plot_confusion_matrix(cm, args.cm)
          
    print("DONE")