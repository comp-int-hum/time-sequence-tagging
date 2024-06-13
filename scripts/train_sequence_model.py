import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import json
import os
from utility import make_dirs, parse_labels
import gzip
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import logging
from models import SequenceTagger, SequenceTaggerWithBahdanauAttention, MulticlassSequenceTaggerWithBahdanauAttention


def get_batch_data(datapoint):
    data = datapoint["sequence_embeds"]
    label = datapoint["labels"]
    metadata = {"id": datapoint["id"],
                "original_text" : datapoint["original_text"],
                "paragraph": datapoint["paragraph"],
                "chapter": datapoint["chapter"],
                "granularity" : datapoint["granularity"],
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
        batch_label = [rnn_utils.pad_sequence([torch.tensor(l) for l in label_class], batch_first=True).to(device) for label_class in zip(*batch_label)]
        data_batches.append(batch_data.to(device))
        label_batches.append(batch_label)
        metadata_batches.append(batch_metadata)
        
    # Open data file
    with gzip.open(filepath, 'r') as gzipfile:
        datapoints = jsonlines.Reader(gzipfile)
        for datapoint in datapoints:
            total_datapoints += 1
            data, label, metadata = get_batch_data(datapoint)
                
            # Append data
            batch_data.append(data)
            batch_label.append(list(zip(*label)))
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

def evaluate(model, batches, metadata, class_labels, device):
    model.eval()
    
    input_data, input_labels = batches
    assert len(metadata) == len(input_data)
    assert len(metadata) == len(input_labels)

    all_preds = []
    all_labels = []
    # all_metadata = [item for sublist in metadata for item in sublist]
    with torch.no_grad():
        for input, labels in zip(input_data, input_labels):
            # If empty batch, pass
            if input.size(0) == 0:
                print("Empty batch")
                continue

            input = input.to(device)
            labels = [l.to(device) for l in labels]

            # Call to forward (possible multi-class outputs)
            outputs = model(input, device = device)
            reshaped_outputs, reshaped_labels = reshape_outputs_labels(outputs, labels, class_labels)
            preds = [torch.argmax(reshaped_output, dim = 1) for reshaped_output in reshaped_outputs]

            all_preds.append([pred.cpu().tolist() for pred in preds])
            all_labels.append(reshaped_labels)
    
    # Calculate final metrics
    all_labels = [sum(sublists, []) for sublists in zip(*all_labels)]
    all_preds = [sum(sublists, []) for sublists in zip(*all_preds)]

    print(f"All labels: {all_labels}")
    print(f"All preds: {all_preds}")

    cms = []
    f1s = []
    accuracies = []
    
    # Calculate metrics across classes
    for all_label, all_pred, class_label in zip(all_labels, all_preds, class_labels):
        labels=list(range(len(class_label)))
        cm = confusion_matrix(y_true=all_label, y_pred=all_pred, labels=labels)
        f1 = f1_score(y_true=all_label, y_pred=all_pred, labels=labels, average=None)
        accuracy = accuracy_score(y_true = all_label, y_pred = all_pred)
        cms.append(cm)
        f1s.append(f1)
        accuracies.append(accuracy)

    return accuracies, f1s, cms, []
  
# Flatten outputs and labels, taking into account multi-class
# List of multiclass, flattened
def reshape_outputs_labels(outputs, labels, classes):
    assert len(classes) == len(outputs)
    outputs = [output.view(-1, len(class_lst)) for class_lst, output in zip(classes, outputs)] # (N, L, num_classes) -> (N x L, num_classes)
    labels = [l.view(-1).cpu().tolist() for l in labels] # NUM_CLASSES x [(N, L) -> (N x L)]
    return outputs, labels


def calculate_accuracy(preds, labels, metadata):
    incorrect_texts = []

    # Get predictions and incorrectly predicted sequences
    predicted_classes = torch.argmax(preds, dim=2)
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
    
def plot_confusion_matrix(cms, label_classes, save_names):
    for cm, label_class, save_name in zip(cms, label_classes, save_names):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = label_class)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        plt.savefig(save_name)
        plt.close(fig)

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
    parser.add_argument("--classes", dest = "classes", type = parse_labels, help = "What the class labels should look like")
    parser.add_argument("--confusion", dest="cm", nargs = "*", help="Name for confusion matrix file")
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    make_dirs(args.model_name)
    make_dirs(args.result)

    print(f"LEN OF CMMMMMMMMMMMMMMMMMM: {args.cm}")
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
    elif "multiclass_sequence_tagger_with_bahdanau_attention" == args.model:
        model = MulticlassSequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, label_classes = args.classes)
    else:
        model = SequenceTagger(input_size = args.emb_dim, num_classes = num_classes)
            
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
                label = [l.to(device) for l in label]
                print(f"Before forward")
                loss, _ = model(input, labels = label, device = device)
                print(f"Done with forward")
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # input_len += label.size(0)
                input_len += 32
            
            print(f"Epoch: {epoch}, Loss: {running_loss / input_len}")
            file.write(f"Epoch: {epoch}, Loss: {running_loss / input_len}\n")
                
            # Eval
            print(f"Test batches: {len(test_batches)}")
            print(f"Test batches metadata: {len(test_metadata)}")
            accuracies, f1s, cms, incorrect_texts = evaluate(model, test_batches, test_metadata, args.classes, device)
            print(f"Type of incorrect_texts: {type(incorrect_texts)}")
            # print(f"Accuracy: {accuracy:.2f}")
            # file.write(f"Accuracy: {accuracy:.2f}\n")
            file.write(f"Accuracies: {accuracies} \n")
            file.write(f"F scores: {f1s} \n")
                
                # if accuracy > best_accuracy:
                #     best_accuracy = accuracy
                #     torch.save(model.state_dict(), args.model_name)
        print("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))
        file.write(f"Train size: {train_size} \n")
        file.write(f"Train positive: {train_positive} \n")
        file.write(f"Test size: {test_size} \n")
        file.write(f"Test positive: {test_positive} \n")
        # file.write(f"Number of incorrect texts: {len(incorrect_texts)}")
        file.write("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))

    with open(args.errors, "w") as errors:
        errors.write(json.dumps(incorrect_texts))
    # print(f"CM: {cm}")
    plot_confusion_matrix(cms, args.classes, args.cm)
          
    print("DONE")