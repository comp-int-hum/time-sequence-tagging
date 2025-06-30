import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
from utility import open_file, make_parent_dirs
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import logging
from tqdm import tqdm
import gzip
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
import texttable as tt
import json
import pickle
import os
import torch
from batch_utils import get_batch, unpad_predictions
from generic_hrnn import GenericHRNN

logger = logging.getLogger("evaluate_model")


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

def apply_metrics(scores, true_labels, metrics, hrnn_layer_names, threshold = 0.5):
    _, num_layers_minus_one = true_labels.shape
    
    layerwise_metrics = []
    
    for l, layer_name in enumerate(hrnn_layer_names):
    
        # Scores and labels
        layer_scores = scores[:, l]
        layer_guesses = (layer_scores > threshold).int().numpy()
        
        layer_labels = true_labels[:, l].numpy()
        
        computed_metrics = []
        
        for metric in metrics:
            computed_metrics.append(
                {
                    "metric_name": metric.__name__,
                    "value": metric(layer_labels, layer_guesses)
                }
            )
        
        layerwise_metrics.append(
            {
                "layer_name": layer_name,
                "layer_num": l,
                "metrics": computed_metrics
            }
        )
        
    return layerwise_metrics
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Input data file")
    parser.add_argument("--model", dest="model", help = "Trained model")
    
    parser.add_argument("--output", dest = "output", help = "Output file for trained model")
    # Params
    parser.add_argument("--batch_size", dest="batch_size", type = int, default=32, help="Batch size")
    parser.add_argument("--threshold", type = float, default = 0.5, help = "Boundary prediction threshold")
    
    args, rest = parser.parse_known_args()
    
    make_parent_dirs(args.output)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    # Get batches
    input_batches = get_batch(args.input, batch_size=args.batch_size, device = device)
    
    with gzip.open(args.input, "rt") as ifd:
        # print(ifd.readline(),  flush = True)
        j = json.loads(ifd.readline())
        emb_dim = len(j["flattened_embeddings"][0])
        num_layers_minus_one = len(j["hierarchical_labels"][0])
        print(f"Num layers - 1: {num_layers_minus_one}")
    
    
    model = GenericHRNN(
        input_size = emb_dim,
        hidden_size = 512,
        num_layers = 3,
        dropout = 0.,
        device = device
    )
    
    model.load_state_dict(torch.load(args.model))
        
    logger.info("%s", model)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    (guesses, golds) = run_model(model, input_batches["inputs"], device = device)
    
    # dev_f1_score = get_f1_score(golds, guesses)
    
    with open(args.output, "wb") as output_file:
        pickle.dump(
            {
                "guesses": guesses,
                "golds": golds,
                "metadata": input_batches["metadata"],
                "lengths": input_batches["lengths"],
                "sentences": input_batches["sentences"]
            },
            output_file
        )