import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import os
from utility import open_file
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import logging
from tqdm import tqdm
import gzip
import json
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from collections import defaultdict
import texttable as tt
from torch.nn.functional import cross_entropy
from generic_hrnn import GenericHRNN
import matplotlib.pyplot as plt

logger = logging.getLogger("train_sequence_model")

def calculate_loss(predictions, teacher_labels):
    """
    predictions: transition_probs (batch_size, seq_len, num_layers - 1)
    teacher_labels: Tensor of true labels (shape: batch_size, seq_len, num_layers)
    """
    # predictions: [batch_size, seq_len, num_layers - 1]
    # predictions = torch.stack(predictions, dim=1)  # (batch_size, sequence_len, num_classes)
    
    
    # batch_size, seq_len, num_classes = predictions.shape
    # assert teacher_labels.shape == (batch_size, seq_len, num_classes), "Mismatch in label shape"
    assert predictions.shape == teacher_labels.shape, "Shape mismatch between predictions and teacher labels"
    
    predictions = predictions.view(-1)  # (batch_size * seq_len, num_classes)
    teacher_labels = teacher_labels.view(-1)         # (batch_size * seq_len)
    
    loss_fn = nn.BCELoss()
    
    # Compute the loss
    loss = loss_fn(predictions, teacher_labels.float())
    return loss

def unpack_data(datapoint):
    """Unpack data from datapoint dict

    Args:
        datapoint (dict): a dictionary representing a sequence from a text and its metadata

    Returns:
        tuple: (embeddings, multiclass labels, metadata)
    """
    # labels = [datapoint["paragraph_labels"], datapoint["chapter_labels"]] if args.boundary_type == "both" else [datapoint["{}_labels".format(args.boundary_type)]]
    # return datapoint.pop("flattened_embeddings"), labels, datapoint
    return datapoint["flattened_embeddings"], datapoint["hierarchical_labels"], datapoint["metadata"]


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
        # print(f"Batch data type: {type(batch_data)}")
        # print(f"Sample batch data: {batch_data}")
        batch_data = rnn_utils.pad_sequence([torch.tensor(d) for d in batch_data], batch_first=True) # [batch_size, seq_len, emb_size]
        print(f"Batch data shape: {batch_data.shape}")
        # print(f"Batch label type: {type(batch_label)}")
        # print(f"Sample batch label: {batch_label}")
        batch_label = rnn_utils.pad_sequence([torch.tensor(l) for l in batch_label], batch_first=True).to(device)
        print(f"Batch label shape: {batch_label.shape}")
        # [batch_size, seq_len, num_layers]
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


def run_model(model, optimizer, batches, device="cpu", is_train=True, teacher_ratio = 0.6):
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
    
    guesses = []
    golds = []
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
            
            out = model(input, teacher_forcing = labels, teacher_ratio = teacher_ratio)
            loss = calculate_loss(out, labels)
            
            golds.append(labels.detach().cpu())
            guesses.append(out.detach().cpu())
            
            if is_train:                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            input_len += input.size(0)
    
    
    
    return (running_loss / input_len), (guesses, golds)
    #         guesses.append(torch.argmax(out, dim=3))
    #         golds.append(labels)
                
    # guesses = torch.concat([torch.reshape(torch.permute(x, (0, 2, 1)), (-1, out.shape[1])) for x in guesses]).cpu()
    # golds = torch.concat([torch.reshape(torch.permute(x, (0, 2, 1)), (-1, out.shape[1])) for x in golds]).cpu()

    # return ((running_loss / input_len), (guesses, golds))

def get_boundary_visualization(boundary_preds, save_path):
    boundaries = boundary_preds.numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(boundaries, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Boundary Prediction Value")
    plt.title("Boundary Visualization")
    plt.xlabel("num_layers - 1")
    plt.ylabel("Sequence Length (seq_len)")
    plt.xticks(range(boundary_preds.shape[1])) # columns = num_layers
    plt.yticks(range(boundary_preds.shape[0])) # rows = seq_len
    plt.show()
    plt.savefig(save_path)

def get_f1_score(guesses, golds):
    golds = torch.cat(golds, dim = 0).int()
    guesses = torch.cat(guesses, dim = 0)
    guesses_thresholded = (guesses > 0.5).int()
    
    flattened_guesses = guesses_thresholded.view(-1).numpy()
    flattened_golds = golds.view(-1).numpy()
    
    print(f"Flattened guesses shape: {flattened_guesses.shape}")
    print(f"Flattened golds shape: {flattened_golds.shape}")
    
    print(f"Flattened_guesses: {flattened_guesses}")
    
    return f1_score(flattened_golds, flattened_guesses)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest = "train", help = "Train file")
    parser.add_argument("--dev", dest = "dev", help = "Dev file")
    parser.add_argument("--test", dest = "test", help = "Test file")
    parser.add_argument("--output", dest = "output", help = "Output file for trained model")
    parser.add_argument("--visualization", dest = "visualization")
    parser.add_argument("--teacher_ratio", type = float, help = "Teacher ratio to use")

    # # Model parameters
    # parser.add_argument("--model", dest="model", choices = ["lstm", "mingru", "minlstm"], help="Type of model, classifier vs sequence_tagger")

    # Training params
    parser.add_argument("--num_epochs", dest="num_epochs", type = int, help="number of epochs to train")
    parser.add_argument("--batch_size", dest="batch_size", type = int, default=32, help="Batch size")
    parser.add_argument("--dropout", dest="dropout", type=float, default = 0.4)
    
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_epochs = args.num_epochs

    best_accuracy = 0
    
    print(f"Num epochs: {args.num_epochs}")

    # task_names = {0 : "paragraph", 1 : "chapter"} if args.boundary_type == "both" else {0 : args.boundary_type}

    
    # Get batches
    train_batches, train_metadata, train_size = get_batch(args.train, batch_size=args.batch_size, device = device)
    dev_batches, dev_metadata, dev_size = get_batch(args.dev, batch_size=args.batch_size, device = device)

    with gzip.open(args.train, "rt") as ifd:
        j = json.loads(ifd.readline())
        emb_dim = len(j["flattened_embeddings"][0])

    # task_sizes = [3, 3] if args.boundary_type == "both" else [3] # three classes (0, 1, 2) for both paragraph and sentence boundaries
    
    model = GenericHRNN(
        input_size = emb_dim,
        hidden_size = 512,
        num_layers = 3,
        dropout = 0.,
        device = device
    )
    # model = SequenceTagger(
    #     task_sizes = task_sizes,
    #     input_size = emb_dim,
    #     model_name = args.model,
    #     dropout=args.dropout
    # )
    
    logger.info("%s", model)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    without_improvement = 0
    
    prev_dev_loss = float("inf")
    best_dev_loss = float("inf")

    train_losses = []
    dev_losses = []
    train_accs, train_f1s, train_cms = [], [], []
    dev_accs, dev_f1s, dev_cms = [], [], []
    
    dev_f1_scores = []
    for epoch in tqdm(range(args.num_epochs), desc = "Epochs"):

        # Training Loop
        train_loss, (train_guesses, train_golds) = run_model(model, optimizer, train_batches, device = device, is_train=True, teacher_ratio = args.teacher_ratio)
        
        dev_loss, (dev_guesses, dev_golds) = run_model(model, None, dev_batches, device = device, is_train = False, teacher_ratio = 0.0)
        
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        
        print(f"Train losss: {train_loss}")
        print(f"Dev loss: {dev_loss}")
        
        dev_f1_score = get_f1_score(dev_golds, dev_guesses)
        dev_f1_scores.append(dev_f1_score)
    
    with open(args.output, "w") as output_file:
        output_file.write(f"Training losses: {train_losses} \n")
        output_file.write(f"Dev losses: {dev_losses} \n")
        output_file.write(f"Dev f1 scores: {dev_f1_scores} \n")
        output_file.write(f"Dev metadata 0: {json.dumps(dev_metadata[0])}")
    
    # with open(args.visualization, "w") as visualization_file:
    get_boundary_visualization(dev_guesses[0][0], save_path = f"{args.visualization}/dev_guess_1")
    get_boundary_visualization(dev_golds[0][0], save_path = f"{args.visualization}/dev_gold_1")
    
    get_boundary_visualization(dev_guesses[0][1], save_path = f"{args.visualization}/dev_guess_2")
    get_boundary_visualization(dev_golds[0][1], save_path = f"{args.visualization}/dev_gold_2")
        
    #     # Dev Loop
    #     dev_loss, (dev_guesses, dev_golds) = run_model(model, None, dev_batches, device = device, is_train=False)

    #     logger.info("Epoch: %d, Train Loss: %.6f", epoch, train_loss)
    #     logger.info("Epoch: %d, Dev Loss: %.6f", epoch, dev_loss)
    #     for task in range(dev_guesses.shape[1]):
    #         score = f1_score(dev_golds[:, task], dev_guesses[:, task], average="macro")
    #         logger.info("Dev score on %s task: %.3f", task_names[task], score)
            
    #     # Save best model based on dev
    #     if dev_loss < best_dev_loss:
    #         best_dev_loss = dev_loss
    #         logger.info("Saving new best model")
    #         torch.save(model.state_dict(), args.output)
    #         without_improvement = 0
    #     else:
    #         without_improvement += 1
    #         logger.info("%d epochs without improvement", without_improvement)

    #     if without_improvement >= 10:
    #         break

    # model.load_state_dict(torch.load(args.output))

    # if args.test:
    #     test_batches, test_metadata, test_size = get_batch(args.test, args.boundary_type, batch_size=args.batch_size, device = device)
    #     test_loss, (test_guesses, test_golds) = run_model(model, None, test_batches, device = device, is_train=False)
    #     for task in range(test_guesses.shape[1]):
    #         score = f1_score(test_golds[:, task], test_guesses[:, task], average="macro")
    #         logger.info("Test score on %s task: %.3f", task_names[task], score)
    # else:
    #     dev_loss, (dev_guesses, dev_golds) = run_model(model, None, dev_batches, device = device, is_train=False)
    #     for task in range(dev_guesses.shape[1]):
    #         score = f1_score(dev_golds[:, task], dev_guesses[:, task], average="macro")
    #         logger.info("Final dev score on %s task: %.3f", task_names[task], score)
