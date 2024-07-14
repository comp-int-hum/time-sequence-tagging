import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import json
import os
from utility import make_dirs, make_dir, parse_labels
import gzip
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import logging
from models import SequenceTagger, SequenceTaggerWithBahdanauAttention, GeneralMulticlassSequenceTaggerWithBahdanauAttention
import pickle
from collections import defaultdict
from tqdm import tqdm

def unpack_data(datapoint):
    data = datapoint.pop("embeddings")
    paragraph_labels = datapoint["paragraph_labels"]
    chapter_labels = datapoint["chapter_labels"]
    metadata = datapoint

    return data, [paragraph_labels, chapter_labels], metadata
    
def get_batch(filepath, batch_size=32, device="cuda"):
    data_batches, label_batches, metadata_batches = [], [], []
    data_batch, label_batch, metadata_batch = [], [], []
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
    with gzip.open(filepath, "r") as gzipfile:
        datapoints = jsonlines.Reader(gzipfile)
        for datapoint in datapoints:
            total_datapoints += 1
            data, label, metadata = unpack_data(datapoint)
                
            # Append data
            data_batch.append(data)
            label_batch.append(label)
            metadata_batch.append(metadata)
            
            # Add batch if batch_sized has been reached
            if len(data_batch) == batch_size:
                append_data(data_batch, label_batch, metadata_batch)
                data_batch, label_batch, metadata_batch = [], [], []
        
        # Add leftover data items to a batch
        if data_batch:
            append_data(data_batch, label_batch, metadata_batch)

    print(f"NUM DATAPOINTS IN FILE {total_datapoints}")
    return (data_batches, label_batches), metadata_batches, positive_datapoints, total_datapoints

def evaluate(model, batches, metadata, class_labels, device):
    model.eval()
    
    input_data, input_labels = batches
    assert len(metadata) == len(input_data)
    assert len(metadata) == len(input_labels)

    all_preds = []
    all_labels = []
    all_pred_values = []
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
            # preds = [torch.argmax(reshaped_output, dim = 1) for reshaped_output in reshaped_outputs]

            # all_preds.append([pred.cpu().tolist() for pred in preds])
            # all_preds.append([
            #     torch.argmax(reshaped_output, dim=1).cpu().tolist() if reshaped_output.numel() > 0 else []
            #     for reshaped_output in reshaped_outputs
            # ])
            # all_pred_values.append([torch.max(reshaped_output, dim=1)[0].cpu().tolist()
            #     if reshaped_output.numel() > 0 else []
            #     for reshaped_output in reshaped_outputs
            # ])
            # all_labels.append(reshaped_labels)
            max_values_indices = [
                torch.max(reshaped_output, dim=1) if reshaped_output.numel() > 0 else (torch.tensor([]), torch.tensor([]))
                for reshaped_output in reshaped_outputs
            ]

            # Separate the values and indices
            max_values = [max_val_ind[0].cpu().tolist() for max_val_ind in max_values_indices]
            indices = [max_val_ind[1].cpu().tolist() for max_val_ind in max_values_indices]

            all_preds.append(indices)
            all_pred_values.append(max_values)
            all_labels.append(reshaped_labels)
    
    # Calculate final metrics
    all_labels = [sum(sublists, []) for sublists in zip(*all_labels)]
    all_preds = [sum(sublists, []) for sublists in zip(*all_preds)]
    all_pred_values = [sum(sublists, []) for sublists in zip(*all_pred_values)]
    all_metadata = [sent for batch in metadata for doc in batch for sent in doc["text"]]

    print(f"length of labels: {len(all_labels[0])}")
    print(f"Length of flattened metadata: {len(all_metadata)}")

    # print(f"All labels: {all_labels}")
    # print(f"All preds: {all_preds}")

    cms = []
    f1s = []
    accuracies = []
    
    # Calculate metrics across classes
    print(f"Length of all labels ({len(all_labels)} - all preds ({len(all_preds)} - class labels {len(class_labels)}))")
    for all_label, all_pred, class_label in zip(all_labels, all_preds, class_labels):
        print(f"CLASS LABEL: {class_label}")
        if class_label:
            print(f"LEGITIMATE CLASS LABEL")
            labels=list(range(len(class_label)))
            cm = confusion_matrix(y_true=all_label, y_pred=all_pred, labels=labels)
            f1 = f1_score(y_true=all_label, y_pred=all_pred, labels=labels, average=None)
            accuracy = accuracy_score(y_true = all_label, y_pred = all_pred)
            cms.append(cm)
            f1s.append(f1)
            accuracies.append(accuracy)
        else:
            print("ILLEGITIMATE LABEL")

    return accuracies, f1s, cms, get_confusion_metadata(all_labels, all_preds, all_pred_values, class_labels, all_metadata)

def get_confusion_metadata(all_labels, all_preds, all_pred_values, class_labels, metadata):
    multiclass_labelled_datapoints = []
    for labels, preds, pred_values, class_label in zip(all_labels, all_preds, all_pred_values, class_labels):
        if class_label:
            
            # Build confusion dictionary
            confusion_datapoints = defaultdict(list)
            for true_label, pred, pred_value, meta in zip(labels, preds, pred_values, metadata):
                true_class_name = class_label[true_label]
                pred_class_name = class_label[pred]
                confusion_datapoints[f"True: {true_class_name} - Pred: {pred_class_name}"].append((pred_value, meta))
                
            # Sort datapoints in each confusion category and only retain 30 most pertinent
            for key in confusion_datapoints:
                confusion_datapoints[key].sort(key=lambda x: x[0])
                
                # Remove duplicate datapoints (although prediction values are based on first seen)
                unique_datapoints = []
                seen = set()
                for x in confusion_datapoints[key]:
                    if x[1] not in seen:
                        unique_datapoints.append(x)
                        seen.add(x[1])
                confusion_datapoints[key] = unique_datapoints
                
                # Limit results to bottom 15 and top 15
                if len(confusion_datapoints[key]) > 30:
                    confusion_datapoints[key] = confusion_datapoints[key][:15] + confusion_datapoints[key][-15:]
                    
            multiclass_labelled_datapoints.append(confusion_datapoints)
    return multiclass_labelled_datapoints



  
# Flatten outputs and labels, taking into account multi-class
# List of multiclass, flattened
def reshape_outputs_labels(outputs, labels, classes):
    assert len(classes) == len(outputs)
    reshaped_outputs = [] # output.view(-1, len(class_lst)) for class_lst, output in zip(classes, outputs)] # (N, L, num_classes) -> (N x L, num_classes)
    reshaped_labels = []
    print(f"CLASSES: {len(classes)} - OUTPUTS: {len(outputs)} - LABELS: {len(labels)}")
    for class_lst, output, label in zip(classes, outputs, labels):
        if class_lst:
            reshaped_outputs.append(output.view(-1, len(class_lst)))
        else:
            reshaped_outputs.append(torch.empty(0, dtype=torch.long))
        if len(class_lst) == 2:
            transformed_label = (label.view(-1) > 0).long()
        else:
            transformed_label = label.view(-1)

        reshaped_labels.append(transformed_label.cpu().tolist())
    return reshaped_outputs, reshaped_labels


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
    
def plot_confusion_matrix(output_dir, cms, label_classes):
    cols = 2
    rows = (len(cms) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize = (10 * cols, 10 * rows))
    axes = axes.flatten()

    label_classes = [labels for labels in label_classes if labels]

    for ax, cm, label_class in zip(axes, cms, label_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = label_class)
        disp.plot(ax=ax, xticks_rotation="vertical")
        ax.set_title("".join(label_class))
    
    for ax in axes[len(cms):]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_confusion_matrices.png"))
    plt.close(fig)

def plot_losses(output_dir, train_losses, dev_losses):
    epochs = list(range(len(train_losses)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label = "Training Loss")
    plt.plot(epochs, dev_losses, label = "Validation Loss")

    plt.title("Training and Dev Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plot_path = os.path.join(output_dir, 'train_and_dev_loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

def print_confusion_metadata(output_dir, confusion_metadata):
    with open(os.path.join(output_dir, "confusion_metadata.txt"), "wt") as output_file:
        for conf_metadata in confusion_metadata:
            for key in conf_metadata.keys():
                output_file.write(f"************************ {key} **************************\n")
                for datapoint in conf_metadata[key]:
                    output_file.write(f"{datapoint[0]} ||| {datapoint[1]}\n")
                    output_file.write("\n")
            output_file.write("#######################################################################\n")
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest = "data", nargs = 3, help = "Train, dev, and test file paths")
    parser.add_argument("--model_save_name", dest="model_name", help="Name of best model")
    parser.add_argument("--visualizations", dest = "visualizations", help = "Output directory path for visualizations and results")
    parser.add_argument("--output_data", dest = "output", help = "Output file for collating and reporting in later steps")

    # Model parameters
    parser.add_argument("--model", dest="model", help="Type of model, classifier vs sequence_tagger")
    parser.add_argument("--emb_dim", dest="emb_dim", type = int, help="size of sentence embedding")
    parser.add_argument("--output_layers", dest = "output_layers", type = int)
    parser.add_argument("--classes", dest = "classes", type = parse_labels, help = "What the class labels should look like")

    # Training params
    parser.add_argument("--num_epochs", dest="epochs", type = int, help="number of epochs to train")
    parser.add_argument("--batch", dest="batch", type = int, default=32, help="Batch size")
    
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    print(f"PARSED LABELS: ****************************************")
    print(args.classes)

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_epochs = args.epochs

    best_accuracy = 0

    num_classes = len(args.classes)
    # Get batches
    train_batches, train_metadata, train_positive, train_size = get_batch(args.data[0], device = device)
    dev_batches, dev_metadata, dev_positive, dev_size = get_batch(args.data[1], batch_size = 1, device = device)
    test_batches, test_metadata, test_positive, test_size = get_batch(args.data[2], batch_size=1, device = device)


    if "sequence_tagger_with_bahdanau_attention" == args.model:
        model = SequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, num_classes = num_classes)
    elif "multiclass_sequence_tagger_with_bahdanau_attention" == args.model:
        model = GeneralMulticlassSequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, label_classes = args.classes, label_class_weights = None, output_layers = args.output_layers, lstm_layers = 1)
    else:
        model = SequenceTagger(input_size = args.emb_dim, label_classes = args.classes, label_class_weights = None, output_layers = args.output_layers, lstm_layers = 1)
            
    model.to(device)

    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, )

    without_improvement = 0
    
    prev_dev_loss = float("inf")
    best_dev_loss = float("inf")
    

    make_dirs(args.model_name)
    print(f"Model name: {args.model_name}")
    make_dir(args.visualizations)
    
    # os.makedirs(os.path.dirname(args.model_name), exist_ok=True)
    # os.makedirs(os.path.dirname(args.visualizations), exist_ok=True)

    model_log = os.path.join(args.visualizations, "model_training_log.txt")

    # make_dirs(model_log)
    with open(model_log, "w") as log_file:

        train_losses = []
        dev_losses = []

        for epoch in tqdm(range(num_epochs), desc = "Epochs"):
            model.train()
            running_train_loss = 0.0
            running_dev_loss = 0.0
            train_input_len = 0
            dev_input_len = 0
            
            # Training Loop
            for train_input, train_label in tqdm(zip(*train_batches), desc = "Training loop"):
                optimizer.zero_grad()
                train_input = train_input.to(device)
                train_label = [l.to(device) for l in train_label]
                train_loss, _ = model(train_input, labels = train_label, device = device)
                train_loss.backward()
                optimizer.step()

                running_train_loss += train_loss.item()
                train_input_len += train_input.size(0)

            # Dev Loop
            with torch.no_grad():
                for dev_input, dev_label in zip(*dev_batches):
                    dev_input = dev_input.to(device)
                    dev_label = [l.to(device) for l in dev_label]
                    dev_loss, _ = model(dev_input, labels = dev_label, device = device)
                    
                    running_dev_loss += dev_loss.item()
                    dev_input_len += dev_input.size(0)

            # Calculate and save losses
            epoch_train_loss = running_train_loss / train_input_len
            epoch_dev_loss = running_dev_loss / dev_input_len
            train_losses.append(epoch_train_loss)
            dev_losses.append(epoch_dev_loss)

            print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss}")
            print(f"Epoch: {epoch}, Dev Loss: {epoch_dev_loss}")
            log_file.write(f"Epoch: {epoch}, Train Loss: {epoch_train_loss} || Dev Loss: {epoch_dev_loss} \n")
            
            # Save best model based on dev
            if epoch_dev_loss < best_dev_loss:
                best_dev_loss = epoch_dev_loss
                torch.save(model.state_dict(), args.model_name)
                
            if abs(epoch_dev_loss - prev_dev_loss) < 0.005:
                without_improvement += 1
            else:
                without_improvement = 0

            prev_dev_loss = epoch_dev_loss

            if without_improvement >= 20:
                break


        # Test
        print(f"Test batches: {len(test_batches)}")
        print(f"Test batches metadata: {len(test_metadata)}")
        model.load_state_dict(torch.load(args.model_name))
        accuracies, f1s, cms, confusion_metadata = evaluate(model, test_batches, test_metadata, args.classes, device)
        log_file.write(f"Test Accuracies: {accuracies} \n")
        log_file.write(f"Test F scores: {f1s} \n")
        log_file.write(f"Train size: {train_size} \n")
        log_file.write(f"Test size: {test_size} \n")

    plot_losses(args.visualizations, train_losses, dev_losses)
    plot_confusion_matrix(args.visualizations, cms, args.classes)
    print_confusion_metadata(args.visualizations, confusion_metadata)

    with open(args.output, "wb") as output_data_file:
        pickle.dump((train_losses, dev_losses, cms, args.classes, accuracies, f1s), output_data_file)
