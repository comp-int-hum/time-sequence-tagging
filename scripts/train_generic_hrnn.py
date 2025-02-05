import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
from utility import open_file, make_dirs
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import logging
from tqdm import tqdm
import gzip
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from collections import defaultdict
import texttable as tt
from generic_hrnn import GenericHRNN
import matplotlib.pyplot as plt
import texttable as tt
import json
import pickle
import os
import torch
import re

logger = logging.getLogger("train_sequence_model")

def sanitize_filename(title, max_length=255):
    sanitized = re.sub(r'[\/:*?"<>|]', '_', title)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized[:max_length]

def calculate_loss(predictions, teacher_labels, layer_weights, balance_pos_neg = None, device = "cpu"):
    """
    predictions: transition_probs (batch_size, seq_len, num_layers - 1)
    teacher_labels: Tensor of true labels (shape: batch_si[[ze, seq_len, num_layers-1)
    layer_weights: List: (num_layers - 1)
    """
    
    layer_weights_tensor = torch.tensor(layer_weights, dtype=predictions.dtype, device = device)
    assert predictions.shape == teacher_labels.shape, "Shape mismatch between predictions and teacher labels"
    batch_size, seq_len, num_layers_minus_one = predictions.shape
    
    predictions = predictions.view(-1, num_layers_minus_one) # (batch_size * seq_len, num_classes)
    teacher_labels = teacher_labels.view(-1, num_layers_minus_one) # (batch_size * seq_len, num_classes)
    
    loss_fn = nn.BCELoss(reduction = "none")

    pos_mask = (teacher_labels == 1).float()
    neg_mask = (teacher_labels == 0).float()

    ele_loss = loss_fn(predictions, teacher_labels.float())
    
    print(f"ele loss shape: {ele_loss.shape}")
    print(f"Layer weight tensor: {layer_weights_tensor.shape}")
    
    weighted_loss = ele_loss * layer_weights_tensor # broadcasting across last dim = num_classes
    
    layerwise_loss = weighted_loss.mean(dim=0)  # (num_layers - 1) layers
    
    if balance_pos_neg:
        assert len(balance_pos_neg) == 2
        pos_neg_weights = torch.tensor(balance_pos_neg, dtype=predictions.dtype, device = device)
        pos_loss_per_layer = (pos_mask * weighted_loss).sum(dim=0) / (pos_mask.sum(dim=0) + 1e-6) * pos_neg_weights[0]
        neg_loss_per_layer = (neg_mask * weighted_loss).sum(dim=0) / (neg_mask.sum(dim=0) + 1e-6) * pos_neg_weights[1]
        
        overall_loss = (pos_loss_per_layer.mean() + neg_loss_per_layer.mean()) / 2
        loss_per_layer = (pos_loss_per_layer + neg_loss_per_layer) / 2  # Loss for each layer
    else:
        overall_loss = weighted_loss.mean()
        loss_per_layer = layerwise_loss

    return overall_loss, loss_per_layer

    # if balance_pos_neg:
    #     pos_loss = (pos_mask * weighted_loss).sum() / (pos_mask.sum() + 1e-6)
    #     neg_loss = (neg_mask * weighted_loss).sum() / (neg_mask.sum() + 1e-6)
    #     return (pos_loss + neg_loss) / 2
    # else:
    #     return weighted_loss.mean()

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
    data_batches, label_batches, metadata_batches, length_batches = [], [], [], []
    data_batch, label_batch, metadata_batch = [], [], []
    
    # Helper function for appending batches and padding if model_type is sequence_tagger
    def append_data(batch_data, batch_label, batch_metadata):
        # print(f"Batch data type: {type(batch_data)}")
        # print(f"Sample batch data: {batch_data}")
        batch_lengths = [len(l) for l in batch_label]
        batch_data = rnn_utils.pad_sequence([torch.tensor(d) for d in batch_data], batch_first=True) # [batch_size, seq_len, emb_size]
        print(f"Batch data shape: {batch_data.shape}")
        batch_label = rnn_utils.pad_sequence([torch.tensor(l) for l in batch_label], batch_first=True).to(device)
        print(f"Batch label shape: {batch_label.shape}")
        # [batch_size, seq_len, num_layers]
        
        data_batches.append(batch_data.to(device))
        label_batches.append(batch_label)
        metadata_batches.append(batch_metadata)
        length_batches.append(batch_lengths)
        
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

    return {
        "inputs": (data_batches, label_batches),
        "metadata": metadata_batches,
        "lengths": length_batches,
        "count": total
    }


def run_model(model, optimizer, batches, layer_weights, balance_pos_neg, device="cpu", is_train=True, teacher_ratio = 0.6):
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
    layer_losses = []
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
            loss, loss_per_layer = calculate_loss(out, labels, layer_weights, balance_pos_neg, device)
            
            print(f"Model out: {out.shape}")
            print(f"Labels: {labels.shape}")
            
            golds.append(labels.detach().cpu())
            guesses.append(out.detach().cpu())
            layer_losses.append(loss_per_layer.detach().cpu())
            
            if is_train:                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            input_len += input.size(0)
    
    
    
    return (running_loss / input_len), torch.stack(layer_losses).mean(dim=0).tolist(), (guesses, golds)

def create_boundary_visualization(boundary_pred, boundary_gold, true_length, save_path):
    """
    boundary_pred: (seq_len, num_layers),
    boundary_gold: (seq_len, num_layers)
    length: true length of sequence
    (columns = 2 * num_layers)
    (rows = seq_len)
    
    """
    
    # Truncate to true length
    boundary_pred = boundary_pred[:true_length, :]
    boundary_gold = boundary_gold[:true_length, :]
    
    seq_len, num_layers = boundary_pred.shape
    
    boundary_pred = boundary_pred.numpy()
    boundary_gold = boundary_gold.numpy()
    
    print(f"Boundary pred: {boundary_pred.shape}")
    print(f"Boundary gold: {boundary_gold.shape}")
    
    print(f"Type of boundary pred: {type(boundary_pred)}")
    print(f"Type of boundary gold: {type(boundary_gold)}")
    
    interleaved_boundaries = np.empty((seq_len, 2 * num_layers))
    interleaved_boundaries[:, 0::2] = boundary_pred # preds on evens
    interleaved_boundaries[:, 1::2] = boundary_gold # golds on odds
    
   
    # interleaved_boundaries[:, 0] = 1
    # interleaved_boundaries[:, 2] = 1
    # interleaved_boundaries[:, 1] = 0
    # interleaved_boundaries[:, 3] = 0
    
    interleaved_boundaries = interleaved_boundaries.T # transpose to (2 * num_layers, seq_len)
    
    print(f"Number of chapter boundaries in gold: {boundary_gold[:, 1].sum()}")
    print(f"Number of chapters in interleaved: {interleaved_boundaries[3, :].sum()}")
    
    plt.figure(figsize=(64, 48))
    plt.imshow(interleaved_boundaries, aspect="auto", cmap="coolwarm", interpolation = "none")
    plt.colorbar(label="Boundary Prediction Value")
    plt.title("Boundary Visualization")
    plt.xlabel("num_layers - 1")
    plt.ylabel("Sequence Length (seq_len)")
    
    tick_labels = [f"Pred{i//2}" if i % 2 == 0 else f"Gold{i//2}" for i in range(2 * num_layers)]
    plt.yticks(range(2 * num_layers), tick_labels)
    plt.xticks(range(seq_len))
    
    plt.show()
    plt.savefig(save_path)

def get_f1_score(guesses, golds):
    f1_scores = []
    num_els = 0
    for guess, gold in zip(guesses, golds):
        batch_size = guess.shape[0]
        
        guesses_thresholded = (guess > 0.5).int()
        
        flattened_guesses = guesses_thresholded.view(-1).numpy()
        flattened_golds = gold.int().view(-1).numpy()
        
        f1_scores.append(batch_size * f1_score(flattened_golds, flattened_guesses))
        num_els += batch_size
    
    return sum(f1_scores) / num_els if num_els > 0 else 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest = "train", help = "Train file")
    parser.add_argument("--dev", dest = "dev", help = "Dev file")
    parser.add_argument("--test", dest = "test", help = "Test file")
    
    parser.add_argument("--output", dest = "output", help = "Output file for trained model")
    parser.add_argument("--results", dest = "results", help = "Results (text) file")
    parser.add_argument("--model", dest="model", help = "Trained model")
    parser.add_argument("--visualization", dest = "visualization")
    parser.add_argument("--vis_num", type = int, help = "Number of samples to visualize")
    
    parser.add_argument("--teacher_ratio", type = float, help = "Teacher ratio to use")
    
    # Training params
    parser.add_argument("--num_epochs", dest="num_epochs", type = int, help="number of epochs to train")
    parser.add_argument("--batch_size", dest="batch_size", type = int, default=32, help="Batch size")
    parser.add_argument("--dropout", dest="dropout", type=float, default = 0.4)
    parser.add_argument("--layer_weights", dest = "layer_weights", nargs = "*", type = float, help = "Weights for different hierarchical layers") # required = False, 
    parser.add_argument("--balance_pos_neg", nargs = 2, type = float, help = "Whether to balance positive and negative examples") # required = False, action = "store_true",
    
    args, rest = parser.parse_known_args()
    
    make_dirs(args.visualization)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.cuda.empty_cache()
    
    print(f"Balance pos neg: {args.balance_pos_neg}")
    print(f"Layer Weights: {args.layer_weights}")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_epochs = args.num_epochs

    best_accuracy = 0

    # Get batches
    train_batches = get_batch(args.train, batch_size=args.batch_size, device = device)
    dev_batches = get_batch(args.dev, batch_size=args.batch_size, device = device)
    
    with gzip.open(args.train, "rt") as ifd:
        j = json.loads(ifd.readline())
        emb_dim = len(j["flattened_embeddings"][0])
        num_layers_minus_one = len(j["hierarchical_labels"][0])
        print(f"Num layeres - 1: {num_layers_minus_one}")
    
    layer_weights = args.layer_weights if args.layer_weights else [1.0] * num_layers_minus_one
    print(f"Layer weights: {layer_weights}")
    
    model = GenericHRNN(
        input_size = emb_dim,
        hidden_size = 512,
        num_layers = 3,
        dropout = 0.,
        device = device
    )
    
    logger.info("%s", model)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    without_improvement = 0
    
    prev_dev_loss = float("inf")
    best_dev_loss = float("inf")

    train_losses = []
    dev_losses = []
    # train_accs, train_f1s, train_cms = [], [], []
    # dev_accs, dev_f1s, dev_cms = [], [], []
    
    best_model = None
    
    result_table = tt.Texttable()
    result_table.set_cols_width([10, 10, 10, 40, 40])
    result_table.set_cols_align(["l", "l", "l", "l", "l"])
    result_table.header(["Epoch", "Train Loss", "Dev loss", "Layer Losses", "Dev F1 Scores"])
    
    train_lls = []
    dev_lls = []
    
    dev_f1_scores = []
    for epoch in tqdm(range(args.num_epochs), desc = "Epochs"):

        # Training Loop
        train_loss, train_layer_losses, (train_guesses, train_golds) = run_model(model, optimizer, train_batches["inputs"], layer_weights, args.balance_pos_neg, device = device, is_train=True, teacher_ratio = args.teacher_ratio)
        
        dev_loss, dev_layer_losses, (dev_guesses, dev_golds) = run_model(model, None, dev_batches["inputs"], layer_weights, args.balance_pos_neg, device = device, is_train = False, teacher_ratio = 0.0)
        
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        
        print(f"Train losss: {train_loss}")
        print(f"Dev loss: {dev_loss}")
        
        dev_f1_score = get_f1_score(dev_golds, dev_guesses)
        dev_f1_scores.append(dev_f1_score)
        
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model = model.state_dict()
            
        result_table.add_row([epoch, train_loss, dev_loss, train_layer_losses, dev_f1_score])
    
    with open(args.results, "w") as results_file:
        results_file.write(result_table.draw())
    
    print(f"Size of dev guesses {len(dev_guesses)}, dev_golds {len(dev_golds)}, metadata: {len(dev_batches['metadata'])}")
    
    for i, (dev_guess, dev_gold, dev_meta, dev_lengths) in enumerate(list(zip(dev_guesses, dev_golds, dev_batches["metadata"], dev_batches["lengths"]))[:args.vis_num]):
        title_author = sanitize_filename(f"{dev_meta[0]['title']}-{dev_meta[0]['author']}")
        vis_path_name = f"{args.visualization}/{title_author}.png"
        os.makedirs(os.path.dirname(vis_path_name), exist_ok=True)
        create_boundary_visualization(dev_guess[0], dev_gold[0], dev_lengths[0], save_path = vis_path_name)
    
    if best_model is not None:
        torch.save(best_model, args.model)
    
    with open(args.output, "wb") as output_file:
        pickle.dump(
            {
                "train_guesses": train_guesses,
                "train_golds": train_golds,
                "dev_guesses": dev_guesses,
                "dev_golds": dev_golds
            },
            output_file
        )
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
