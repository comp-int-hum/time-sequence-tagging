import argparse
import json
import os
from utility import make_parent_dirs_for_files
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import torch
import re
import numpy as np
import pickle
import torch.nn.utils.rnn as rnn_utils
from batch_utils import unbatch

def sanitize_filename(title, max_length=255):
    sanitized = re.sub(r'[\/:*?"<>|]', '_', title)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized[:max_length]

def build_visualization_output_name(root_dir, title, author):
    title_author = sanitize_filename(f"{title}-{author}")
    vis_path_name = f"{root_dir}/{title_author}.png"
    os.makedirs(os.path.dirname(vis_path_name), exist_ok=True)
    return vis_path_name

def build_text_name(title, author):
    return sanitize_filename(f"{title}-{author}")
    
def create_boundary_visualization(boundary_pred, boundary_gold, true_length, text_name, save_path):
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
    
    # print(f"Boundary pred: {boundary_pred.shape}")
    # print(f"Boundary gold: {boundary_gold.shape}")
    
    # print(f"Type of boundary pred: {type(boundary_pred)}")
    # print(f"Type of boundary gold: {type(boundary_gold)}")
    
    interleaved_boundaries = np.empty((seq_len, 2 * num_layers))
    interleaved_boundaries[:, 0::2] = boundary_pred # preds on evens
    interleaved_boundaries[:, 1::2] = boundary_gold # golds on odds
    
   
    # interleaved_boundaries[:, 0] = 1
    # interleaved_boundaries[:, 2] = 1
    # interleaved_boundaries[:, 1] = 0
    # interleaved_boundaries[:, 3] = 0
    
    interleaved_boundaries = interleaved_boundaries.T # transpose to (2 * num_layers, seq_len)
    
    # print(f"Number of chapter boundaries in gold: {boundary_gold[:, 1].sum()}")
    # print(f"Number of chapters in interleaved: {interleaved_boundaries[3, :].sum()}")
    
    plt.figure(figsize=(64, 48))
    plt.imshow(interleaved_boundaries, aspect="auto", cmap="coolwarm", interpolation = "none")
    plt.colorbar(label="Boundary Prediction Value")
    plt.title(f"Boundary Visualization - {text_name}")
    plt.xlabel("Sequence Length (sentences)")
    plt.ylabel("Hierarchical Layers")
    
    tick_labels = [f"Pred{i//2}" if i % 2 == 0 else f"Gold{i//2}" for i in range(2 * num_layers)]
    plt.yticks(range(2 * num_layers), tick_labels)
    plt.xticks(range(seq_len))
    
    plt.show()
    plt.savefig(save_path)

def construct_visualizations(batched_model_outputs, hrnn_visualization_paths):
    vis_count = 0
    vis_total = len(args.hrnn_visualizations)
    for (guess_batch, gold_batch, meta_batch, length_batch) in zip(batched_model_outputs["guesses"],
                                                                   batched_model_outputs["golds"],
                                                                   batched_model_outputs["metadata"],
                                                                   batched_model_outputs["lengths"]):
        for (guess, gold, meta, length) in zip(guess_batch, gold_batch, meta_batch, length_batch):
            if vis_count >= vis_total:
                return
            
            text_name = build_text_name(meta['title'], meta['author'])
            create_boundary_visualization(guess, gold, length, text_name, save_path = hrnn_visualization_paths[vis_count])
            vis_count += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Filepath for data containing training outputs")
    parser.add_argument("--hrnn_visualizations", dest = "hrnn_visualizations", nargs = "+", help = "HRNN visualizations")
    parser.add_argument("--hrnn_layer_names", dest = "hrnn_layer_names", nargs = "+", default = ["paragraphs", "chapters"], help = "Names of hierarchical layers in model")
    parser.add_argument("--threshold", type = float, default = 0.5, help = "Boundary decision threshold")
    args = parser.parse_args()
    
    make_parent_dirs_for_files(args.hrnn_visualizations)
    print(f"HRNN Visualizations: {args.hrnn_visualizations}")
    with open(args.input, "rb") as input_file:
        batched_model_outputs = pickle.load(input_file)
        seq_len, num_layers_minus_one = batched_model_outputs["guesses"][0][0].shape
    
    assert len(args.hrnn_layer_names) == num_layers_minus_one, "Number of hierarchical layer names provided does not match layers in guesses"
    
    predictions = unbatch(batched_model_outputs)
    
    print(type(predictions["cumulative_lengths"]))
    
    construct_visualizations(batched_model_outputs, args.hrnn_visualizations)
        
    