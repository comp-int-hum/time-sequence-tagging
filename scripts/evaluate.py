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
import matplotlib.pyplot as plt

logger = logging.getLogger("evaluate_model")


def unpack_data(datapoint):
    """Unpack data from datapoint dict

    Args:
        datapoint (dict): a dictionary representing a sequence from a text and its metadata

    Returns:
        tuple: (embeddings, multiclass labels, metadata)
    """
    # labels = [datapoint["paragraph_labels"], datapoint["chapter_labels"]] if args.boundary_type == "both" else [datapoint["{}_labels".format(args.boundary_type)]]
    # return datapoint.pop("flattened_embeddings"), labels, datapoint
    return datapoint["flattened_embeddings"], datapoint["hierarchical_labels"], datapoint["metadata"], datapoint["flattened_sentences"]


def get_batch(filepath, batch_size=32, device="cpu"):
    """Create batches based on file path and batch_size

    Args:
        filepath (str): filepath to data
        batch_size (int, optional): Batch size for model training. Defaults to 32.
        device (str, optional): Device to move tensors to. Defaults to "cpu".

    Returns:
        tuple: (data_batches, label_batches), metadata_batches
    """
    data_batches, label_batches, metadata_batches, length_batches, sentence_batches = [], [], [], []
    data_batch, label_batch, metadata_batc, sentence_batch = [], [], []
    
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
        sentence_batches.append(sentence_batch)
        
    # Open data file
    total = 0
    with open_file(filepath, "r") as source_file, jsonlines.Reader(source_file) as datapoints:
        for i, datapoint in enumerate(datapoints):
            total += 1
            data, label, metadata, sentences = unpack_data(datapoint)
            
            # Append data
            data_batch.append(data)
            label_batch.append(label)
            metadata_batch.append(metadata)
            sentence_batch.append(sentences)
            
            # Add batch if batch_sized has been reached
            if len(data_batch) == batch_size:
                append_data(data_batch, label_batch, metadata_batch, sentence_batch)
                data_batch, label_batch, metadata_batch, sentence_batch = [], [], [], []
        
        # Add leftover data items to a batch
        if data_batch:
            append_data(data_batch, label_batch, metadata_batch, sentence_batch)

    return {
        "inputs": (data_batches, label_batches),
        "sentences": sentence_batches,
        "metadata": metadata_batches,
        "lengths": length_batches,
        "count": total
    }



def run_model(model, batches, device="cpu"):
    """Evaluate the model given batches and return predictions and ground-truth labels.

    Args:
        model (nn.Module):
        batches (tuple): (input_data, input_labels)
        device:
    Returns:
        tuple: (guesses, golds) — predictions and ground-truth labels.
    """
    model.eval()  # Ensure model is in evaluation mode
    guesses = []
    golds = []

    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for input, labels in tqdm(zip(*batches), desc="Prediction loop"):
            if input.size(0) == 0:
                print("Empty batch")
                continue

            input = input.to(device)
            # labels = labels.to(device)

            out = model(input, teacher_forcing=None, teacher_ratio=0.0)

            golds.append(labels) # .detach().cpu())
            guesses.append(out.detach().cpu())
    
    return guesses, golds


def get_f1_score(guesses, golds, threshold = 0.5):
    
    f1_scores = []
    num_els = 0
    for guess, gold in zip(guesses, golds):
        batch_size = guess.shape[0]
        
        guesses_thresholded = (guess > threshold).int()
        
        flattened_guesses = guesses_thresholded.view(-1).numpy()
        flattened_golds = gold.int().view(-1).numpy()
        
        f1_scores.append(batch_size * f1_score(flattened_golds, flattened_guesses))
        num_els += batch_size
    
    return sum(f1_scores) / num_els if num_els > 0 else 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Input data file")
    parser.add_argument("--model", dest="model", help = "Trained model")
    
    parser.add_argument("--output", dest = "output", help = "Output file for trained model")
    # Params
    parser.add_argument("--batch_size", dest="batch_size", type = int, default=32, help="Batch size")
    parser.add_argument("--threshold", type = float, default = 0.5, help = "Boundary prediction threshold")
    
    args, rest = parser.parse_known_args()
    
    make_dirs(args.output)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    # Get batches
    input_batches = get_batch(args.input, batch_size=args.batch_size, device = device)
    
    with gzip.open(args.input, "rt") as ifd:
        print(ifd.readline(),  flush = True)
        j = json.loads(ifd.readline())
        emb_dim = len(j["flattened_embeddings"][0])
        num_layers_minus_one = len(j["hierarchical_labels"][0])
        print(f"Num layers - 1: {num_layers_minus_one}")
    
    
    model = torch.load(args.model)
    logger.info("%s", model)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    (guesses, golds) = run_model(model, input_batches["inputs"], device="cpu")
    
    dev_f1_score = get_f1_score(golds, guesses)
    
    with open(args.output, "wb") as output_file:
        pickle.dump(
            {
                "guesses": guesses,
                "golds": golds,
                "metadata": input_batches["metadata"],
                "lengths": input_batches["lengths"],
            },
            output_file
        )