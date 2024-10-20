import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import os
from utility import make_dirs, make_dir, parse_labels, open_file
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import logging
from models import SequenceTagger, SequenceTaggerWithBahdanauAttention, GeneralMulticlassSequenceTaggerWithBahdanauAttention
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from collections import defaultdict
import texttable as tt

def unpack_data(datapoint):
    """Unpack data from datapoint dict

    Args:
        datapoint (dict): a dictionary representing a sequence from a text and its metadata

    Returns:
        tuple: (embeddings, multiclass labels, metadata)
    """    
    return datapoint.pop("flattened_embeddings"), [datapoint["paragraph_labels"], datapoint["chapter_labels"]], datapoint

def get_batch(filepath, batch_size=32, device="cpu"):
    """Create batches based on file path and batch_size

    Args:
        filepath (str): filepath to data
        batch_size (int, optional): Batch size for model training. Defaults to 32.
        device (str, optional): Device to move tensors to. Defaults to "cpu".

    Returns:
        tuple: (data_batches, label_batches), metadata_batches
    """    
    data_batches, label_batches, metadata_batches = [], [], []
    data_batch, label_batch, metadata_batch = [], [], []
    
    # Helper function for appending batches and padding if model_type is sequence_tagger
    def append_data(batch_data, batch_label, batch_metadata):
        batch_data = rnn_utils.pad_sequence([torch.tensor(d) for d in batch_data], batch_first=True)
        batch_label = [rnn_utils.pad_sequence([torch.tensor(l) for l in label_class], batch_first=True).to(device) for label_class in zip(*batch_label)]
        data_batches.append(batch_data.to(device))
        label_batches.append(batch_label)
        metadata_batches.append(batch_metadata)
        
    # Open data file
    total = 0
    with open_file(filepath, "r") as source_file, jsonlines.Reader(source_file) as datapoints:
        for i, datapoint in enumerate(datapoints):
            total += 1
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

    return (data_batches, label_batches), metadata_batches, total

def run_model(model, optimizer, batches, device="cpu", is_train=True):
    """Evaluate model given batches, metadata, and class labels

    Args:
        model (): sequence tagging model
        batches (tuple): (input_data, input_label) where each is a list of lists of tensors
        metadata (list): each element is a batch of dict items representing the metadata for the datapoints
        class_labels (list): where each element is a list of different class labels (e.g. none, par_start, par_end)
        device (): device on which to place model

    Returns:
        _type_: _description_
    """
    if is_train:
        model.train()
    else:
        model.eval()
    
    tasks_preds = []
    tasks_labels = []
    tasks_pred_scores = []
    running_loss = 0
    input_len = 0
    
    with torch.set_grad_enabled(is_train):
        for input, labels in tqdm(zip(*batches), desc = "Prediction loop"):
            
            if is_train:
                optimizer.zero_grad()
                
            # If empty batch, pass
            if input.size(0) == 0:
                print("Empty batch")
                continue
            
            # Move to device
            input = input.to(device)
            labels = [l.to(device) for l in labels]

            # Call  to forward (possible multi-class outputs)
            batch_loss, (flattened_outputs, flattened_labels) = model(input, device = device, labels = labels, flatten = True)
            
            if is_train:
                batch_loss.backward()
                optimizer.step()
            
            running_loss += batch_loss.item()
            input_len += input.size(0)
            
            max_values_indices = [
                torch.max(flattened_output, dim=1) if flattened_output.numel() > 0 else (torch.tensor([]), torch.tensor([]))
                for flattened_output in flattened_outputs
            ]

            # Separate the values and indices
            max_values = [max_val_ind[0].cpu().tolist() for max_val_ind in max_values_indices]
            indices = [max_val_ind[1].cpu().tolist() for max_val_ind in max_values_indices]
            labels_cpu = [label.cpu().tolist() for label in flattened_labels]

            tasks_preds.append(indices)
            tasks_pred_scores.append(max_values)
            tasks_labels.append(labels_cpu)
    
    # Calculate final metrics
    tasks_labels = [sum(sublists, []) for sublists in zip(*tasks_labels)]
    tasks_preds = [sum(sublists, []) for sublists in zip(*tasks_preds)]
    tasks_pred_scores = [sum(sublists, []) for sublists in zip(*tasks_pred_scores)]
    
    return (running_loss / input_len), (tasks_labels, tasks_preds, tasks_pred_scores)
    

def get_task_metrics(task_labels, task_preds, label_groups, metrics):
    task_metrics = {metric["func"].__name__: [] for metric in metrics}
    
    for true_labels, predicted_labels, class_labels in zip(task_labels, task_preds, label_groups):
        if class_labels:
            label_indices = list(range(len(class_labels)))
            
            for metric in metrics:
                if "labels" in metric["kwargs"] and not metric["kwargs"]["labels"]:
                    metric["kwargs"]["labels"] = label_indices
                print(f"Metric func: {metric['func']}")
                metric_result = metric["func"](y_true=true_labels, y_pred=predicted_labels, **metric["kwargs"])
                task_metrics[metric["func"].__name__].append(metric_result)
    
    return task_metrics

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
    
def plot_accs(output_dir, tasks_accuracies, task_labels, name):
    epochs = list(range(len(tasks_accuracies)))
    plt.figure(figsize=(10, 6))
    
    task_labels = [task_label for task_label in task_labels if task_label]
    
    for task_acc, task_label in zip(list(zip(*tasks_accuracies)), task_labels):
        plt.plot(epochs, task_acc, label = f"{task_label} accuracy")

    plt.title("Task accuracies vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plot_path = os.path.join(output_dir, f'{name}_accuracies_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
def plot_f1s(output_dir, tasks_f1s, task_labels, name):
    epochs = list(range(len(tasks_f1s)))
    plt.figure(figsize=(10, 6))
    
    task_labels = [task_label for task_label in task_labels if task_label]
    
    for task_acc, task_label in zip(list(zip(*tasks_f1s)), task_labels):
        print(f"task f1: {task_acc}")
        plt.plot(epochs, task_acc, label = f"{task_label} f1s")

    plt.title("Task f1s vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plot_path = os.path.join(output_dir, f'{name}_f1s_plot.png')
    plt.savefig(plot_path)
    plt.close()

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

def print_confusion_metadata(output_dir, confusion_metadata):
    with open(os.path.join(output_dir, "confusion_metadata.txt"), "wt") as output_file:
        for conf_metadata in confusion_metadata:
            for key in conf_metadata.keys():
                output_file.write(f"************************ {key} **************************\n")
                for actual, predicted in conf_metadata[key]:
                    output_file.write(f"{actual} ||| {predicted}\n")
                    output_file.write("\n")
            output_file.write("#######################################################################\n")

# all_metadata = [sent for batch in metadata for doc in batch for sent in doc["flattened_sentences"]]
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
    parser.add_argument("--dropout", dest="dropout", type=float, default = 0.4)
    
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_epochs = args.epochs

    best_accuracy = 0

    num_classes = len(args.classes)
    
    # Get batches
    train_batches, train_metadata, train_size = get_batch(args.data[0], device = device)
    dev_batches, dev_metadata, dev_size = get_batch(args.data[1], batch_size = 1, device = device)
    test_batches, test_metadata, test_size = get_batch(args.data[2], batch_size=1, device = device)


    # Set models
    if "sequence_tagger_with_bahdanau_attention" == args.model:
        model = SequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, num_classes = num_classes)
    elif "multiclass_sequence_tagger_with_bahdanau_attention" == args.model:
        model = GeneralMulticlassSequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, label_classes = args.classes, label_class_weights = None, output_layers = args.output_layers, lstm_layers = 1)
    else:
        model = SequenceTagger(input_size = args.emb_dim, label_classes = args.classes, label_class_weights = None, output_layers = args.output_layers, lstm_layers = 1, dropout=args.dropout)
            
    model.to(device)

    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    without_improvement = 0
    
    prev_dev_loss = float("inf")
    best_dev_loss = float("inf")
    

    make_dirs(args.model_name)
    print(f"Model name: {args.model_name}")
    make_dir(args.visualizations)

    model_log = os.path.join(args.visualizations, "model_training_log.txt")
    
    training_metrics_args = [
        {   "func": accuracy_score,
            "kwargs": {},
        },
        {
            "func": f1_score,
            "kwargs": {"average": None, "labels": None, "zero_division": 0.0},
        },
    ]
    
    test_metrics_args = [
        {   "func": accuracy_score,
            "kwargs": {},
        },
        {
            "func": f1_score,
            "kwargs": {"average": None, "labels": None, "zero_division": 0.0},
        },
        {
            "func": confusion_matrix,
            "kwargs": {"labels": None},
        },
    ]


    with open(model_log, "w") as log_file:

        train_losses = []
        dev_losses = []
        train_accs, train_f1s, train_cms = [], [], []
        dev_accs, dev_f1s, dev_cms = [], [], []

        for epoch in tqdm(range(num_epochs), desc = "Epochs"):
            
            # Training Loop
            epoch_train_loss, (train_labels, train_preds, train_pred_scores) = run_model(model, optimizer, train_batches, device = device, is_train=True)
            train_metrics = get_task_metrics(train_labels, train_preds, args.classes, training_metrics_args)
            
            # Append train metrics
            train_accs.append(train_metrics["accuracy_score"])
            train_f1s.append(train_metrics["f1_score"])

            # Dev Loop
            epoch_dev_loss, (dev_labels, dev_preds, dev_pred_scores) = run_model(model, None, dev_batches, device = device, is_train=False)
            dev_metrics = get_task_metrics(dev_labels, dev_preds, args.classes, training_metrics_args)
            
            # Append dev metrics
            dev_accs.append(dev_metrics["accuracy_score"])
            dev_f1s.append(dev_metrics["f1_score"])

            # Calculate and save losses
            train_losses.append(epoch_train_loss)
            dev_losses.append(epoch_dev_loss)

            print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss}")
            print(f"Epoch: {epoch}, Dev Loss: {epoch_dev_loss}")
            log_file.write(f"Epoch: {epoch}, Train Loss: {epoch_train_loss} || Dev Loss: {epoch_dev_loss} \n")
            
            # Save best model based on dev
            if epoch_dev_loss < best_dev_loss:
                best_dev_loss = epoch_dev_loss
                torch.save(model.state_dict(), args.model_name)
                
            if abs(epoch_dev_loss - prev_dev_loss) < 0.0005:
                without_improvement += 1
            else:
                without_improvement = 0

            prev_dev_loss = epoch_dev_loss

            if without_improvement >= 20:
                break


        print(f"Test batches: {len(test_batches)}")
        print(f"Test batches metadata: {len(test_metadata)}")
        model.load_state_dict(torch.load(args.model_name))
        
        # Test Loop
        test_loss, (test_labels, test_preds, test_pred_scores) = run_model(model, None, test_batches, device = device, is_train=False)
        test_metrics = get_task_metrics(test_labels, test_preds, args.classes, test_metrics_args)
        
        stats_table = tt.Texttable()
        stats_table.set_cols_width([15, 15])
        stats_table.set_cols_align(["l", "l"])
        stats_table.header(["Statistic Name", "Value"])
        stats_table.add_row(["Test Accuracies", ", ".join(map(str,test_metrics["accuracy_score"]))])
        stats_table.add_row(["Test F scores", ", ".join(map(str,test_metrics["f1_score"]))])
        stats_table.add_row(["Train size", train_size])
        stats_table.add_row(["Dev size", dev_size])
        stats_table.add_row(["Test size", test_size])
        log_file.write(stats_table.draw())

    plot_losses(args.visualizations, train_losses, dev_losses)
    plot_accs(args.visualizations, train_accs, args.classes, name = "train")
    plot_accs(args.visualizations, dev_accs, args.classes, name = "dev")
    plot_f1s(args.visualizations, train_f1s, args.classes, name = "train")
    plot_f1s(args.visualizations, dev_f1s, args.classes, name = "dev")
    plot_confusion_matrix(args.visualizations, test_metrics["confusion_matrix"], args.classes)
    
    all_metadata = [sent for batch in test_metadata for doc in batch for sent in doc["flattened_sentences"]]
    confusion_metadata = get_confusion_metadata(test_labels, test_preds, test_pred_scores, args.classes, all_metadata)
    print_confusion_metadata(args.visualizations, confusion_metadata)

    with open(args.output, "wb") as output_data_file:
        pickle.dump(
            {
                "train_losses": train_losses, 
                "dev_losses": dev_losses,
                "train_accs" : train_accs,
                "dev_accs": dev_accs,
                "train_f1s": train_f1s,
                "dev_f1s": dev_f1s,
                "test_labels": test_labels,
                "test_preds": test_preds,
                "test_pred_scores": test_pred_scores,
                "metadata": all_metadata,
                "classes": args.classes
            }, 
            output_data_file
        )

# def get_task_metrics(task_labels, task_preds, label_groups):
#     task_accuracies, task_f1s, task_cms = [], [], []
    
#     # Calculate metrics across classes
#     for all_label, all_pred, class_labels in zip(task_labels, task_preds, label_groups):
#         if class_labels:
#             labels=list(range(len(class_labels)))
#             cm = confusion_matrix(y_true=all_label, y_pred=all_pred, labels=labels)
#             f1 = f1_score(y_true=all_label, y_pred=all_pred, labels=labels, average=None)
#             accuracy = accuracy_score(y_true = all_label, y_pred = all_pred)
#             task_cms.append(cm)
#             task_f1s.append(f1)
#             task_accuracies.append(accuracy)
#         else:
#             print("Empty label")
    
#     return task_accuracies, task_f1s, task_cms