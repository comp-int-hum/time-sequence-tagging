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
import gzip
import json
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from collections import defaultdict
import texttable as tt
from torch.nn.functional import cross_entropy

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
            labels = torch.stack([l.to(device) for l in labels], 1)

            # Call  to forward (possible multi-class outputs)
            #batch_loss, (flattened_outputs, flattened_labels) =
            out = model(input, device = device)
            loss = cross_entropy(torch.permute(out, (0, 3, 1, 2)), labels)
            
            if is_train:
                
                loss.backward()
                optimizer.step()
            #continue
            #print(out.shape)
            #print(labels.shape)
            
            #sys.exit()
            #if is_train:
            #    batch_loss.backward()
            #    optimizer.step()
            
            running_loss += loss.item()
            input_len += input.size(0)
            
            #max_values_indices = [
            #    torch.max(flattened_output, dim=1) if flattened_output.numel() > 0 else (torch.tensor([]), torch.tensor([]))
            #    for flattened_output in flattened_outputs
            #]

            # Separate the values and indices
            #max_values = [max_val_ind[0].cpu().tolist() for max_val_ind in max_values_indices]
            #indices = [max_val_ind[1].cpu().tolist() for max_val_ind in max_values_indices]
            #labels_cpu = [label.cpu().tolist() for label in flattened_labels]

            #tasks_preds.append(indices)
            #tasks_pred_scores.append(max_values)
            #tasks_labels.append(labels_cpu)
    
    # Calculate final metrics
    #tasks_labels = [sum(sublists, []) for sublists in zip(*tasks_labels)]
    #tasks_preds = [sum(sublists, []) for sublists in zip(*tasks_preds)]
    #tasks_pred_scores = [sum(sublists, []) for sublists in zip(*tasks_pred_scores)]
    
    return (running_loss / input_len) #, (tasks_labels, tasks_preds, tasks_pred_scores)
    

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


# all_metadata = [sent for batch in metadata for doc in batch for sent in doc["flattened_sentences"]]
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest = "train", help = "Train file")
    parser.add_argument("--dev", dest = "dev", help = "Dev file")
    parser.add_argument("--test", dest = "test", help = "Test file")
    #parser.add_argument("--model_save_name", dest="model_name", help="Name of best model")
    #parser.add_argument("--visualizations", dest = "visualizations", help = "Output directory path for visualizations and results")
    #parser.add_argument("--output_data", dest = "output", help = "Output file for collating and reporting in later steps")
    parser.add_argument("--output", dest = "output", help = "Output file")

    # Model parameters
    parser.add_argument("--model", dest="model", help="Type of model, classifier vs sequence_tagger")
    #parser.add_argument("--emb_dim", dest="emb_dim", type = int, help="size of sentence embedding")
    #parser.add_argument("--output_layers", dest = "output_layers", type = int)
    #parser.add_argument("--classes", dest = "classes", type = parse_labels, help = "What the class labels should look like")

    # Training params
    parser.add_argument("--num_epochs", dest="epochs", type = int, help="number of epochs to train")
    parser.add_argument("--batch_size", dest="batch_size", type = int, default=32, help="Batch size")
    parser.add_argument("--dropout", dest="dropout", type=float, default = 0.4)
    parser.add_argument("--boundary_type", dest="boundary_type", default = "chapter")
    
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_epochs = args.epochs

    best_accuracy = 0

    #num_classes = len(args.classes)
    
    # Get batches
    train_batches, train_metadata, train_size = get_batch(args.train, device = device)
    dev_batches, dev_metadata, dev_size = get_batch(args.dev, batch_size = 1, device = device)
    test_batches, test_metadata, test_size = get_batch(args.test, batch_size=1, device = device)


    # Set models
    #if "sequence_tagger_with_bahdanau_attention" == args.model:
    #    model = SequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, num_classes = num_classes)
    #elif "multiclass_sequence_tagger_with_bahdanau_attention" == args.model:
    #    model = GeneralMulticlassSequenceTaggerWithBahdanauAttention(input_size = args.emb_dim, label_classes = args.classes, label_class_weights = None, output_layers = args.output_layers, lstm_layers = 1)
    #else:

    with gzip.open(args.train, "rt") as ifd:
        j = json.loads(ifd.readline())
        emb_dim = len(j["flattened_embeddings"][0])

    if args.boundary_type in ["paragraph", "chapter"]:
        num_classes = 3
    else:
        raise NotImplemented() # some combinations make sense, others don't
        
    
    model = SequenceTagger(
        task_sizes = [3, 3],
        lstm_input_size = emb_dim,
        #label_classes = num_classes,
        #label_class_weights = None,
        #output_layers = args.output_layers,
        #lstm_layers = 2,
        #dropout=args.dropout
    )
    print(model)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    without_improvement = 0
    
    prev_dev_loss = float("inf")
    best_dev_loss = float("inf")

    train_losses = []
    dev_losses = []
    train_accs, train_f1s, train_cms = [], [], []
    dev_accs, dev_f1s, dev_cms = [], [], []

    for epoch in tqdm(range(num_epochs), desc = "Epochs"):

        # Training Loop
        #epoch_train_loss, (train_labels, train_preds, train_pred_scores) =
        train_loss = run_model(model, optimizer, train_batches, device = device, is_train=True)

        # Dev Loop
        #epoch_dev_loss, (dev_labels, dev_preds, dev_pred_scores) =
        dev_loss = run_model(model, None, dev_batches, device = device, is_train=False)

        # Calculate and save losses
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        print(f"Epoch: {epoch}, Train Loss: {train_loss}")
        print(f"Epoch: {epoch}, Dev Loss: {dev_loss}")

        # Save best model based on dev
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), args.output)

        if abs(dev_loss - prev_dev_loss) < 0.0005:
            without_improvement += 1
        else:
            without_improvement = 0

        prev_dev_loss = dev_loss

        if without_improvement >= 20:
            break

    model.load_state_dict(torch.load(args.output))
    #test_loss, (test_labels, test_preds, test_pred_scores) =
    test_loss = run_model(model, None, test_batches, device = device, is_train=False)
    print(test_loss)
