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


class BasicClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        print(f"Input size: {input_size}")
        self.fc1 = nn.Linear(input_size, 256) # batch_first, H_in
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in")
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in")
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        nn.init.kaiming_normal_(self.fc3.weight, mode="fan_in")
        self.output = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x = self.fc1(data)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def get_batch_data(datapoint):
    data = datapoint["segment"]
    label = torch.tensor(datapoint["ground_truth"])
    metadata = {"id" : datapoint["id"],
                "first_ch" : datapoint["first_ch_passage"],
                "second_ch" : datapoint["second_ch_passage"],
                "ground_truth": datapoint["ground_truth"]}

    return data, label, metadata


def get_batch(filepath, batch_size=32, device="cuda"):
    data_batches, label_batches, metadata_batches = [], [], []
    batch_data, batch_label, batch_metadata = [], [], []
    total_datapoints = 0
    positive_datapoints = 0
  
    def append_data(batch_data, batch_label, batch_metadata):
        data_batches.append(batch_data.to(device))
        label_batches.append(batch_label.to(device))
        metadata_batches.append(batch_metadata)
        
    # Open data file
    with gzip.open(filepath, "r") as gzipfile:
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
            
            correct, curr_total, incorrect = calculate_accuracy(output, label, meta)
            
            preds = torch.argmax(output, dim = 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
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


def calculate_accuracy(output, labels, metadata):
    incorrect_texts = []
    predicted_classes = torch.argmax(output, dim=1) # (N, num_classes)
    correct_predictions = (predicted_classes == labels)
    num_correct = (correct_predictions).sum().item()

    for i, correct in enumerate(correct_predictions):
        if not correct:
            incorrect_texts.append(metadata[i])
    total = labels.size(0)
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
    train_batches, train_metadata, train_positive, train_size = get_batch(args.test, device = device)
    test_batches, test_metadata, test_positive, test_size = get_batch(args.train, batch_size=1, device = device)

    model = BasicClassifier(input_size = args.emb_dim, num_classes=num_classes)
            
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