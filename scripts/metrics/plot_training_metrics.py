import argparse
import json
import os
from utility import make_parent_dirs_for_files
import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_loss_curves(train_loss, dev_loss, save_path):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="Training Loss", marker="o")
    plt.plot(epochs, dev_loss, label="Validation Loss", marker="o")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Dev Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def plot_layer_losses(train_layer_losses, dev_layer_losses, hrnn_layer_names, save_paths):
    train_layer_losses = np.array(train_layer_losses)  # (num_epochs, num_layers)
    dev_layer_losses = np.array(dev_layer_losses)  # (num_epochs, num_layers)
    
    epochs = range(1, len(train_layer_losses) + 1)

    for lnum, (layer_name, save_path) in enumerate(zip(hrnn_layer_names, save_paths)):
        plt.figure(figsize=(8, 6))
        
        plt.plot(epochs, train_layer_losses[:, lnum], label="Train Loss", marker="o", linestyle="-", linewidth=2)
        plt.plot(epochs, dev_layer_losses[:, lnum], label="Dev Loss", marker="s", linestyle="-", linewidth=2)

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve for {layer_name}")
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Pickle file with training stats")
    parser.add_argument("--loss_curves", dest = "loss_curves", help = "Output file for loss curves")
    parser.add_argument("--layer_loss_curves", dest = "layer_loss_curves", nargs = "+", help = "Output losses over epochs by layer")
    parser.add_argument("--hrnn_layer_names", dest = "hrnn_layer_names", nargs = "+", default = ["paragraphs", "chapters"], help = "Names of hierarchical layers in model")
    args = parser.parse_args()
    
    make_parent_dirs_for_files([args.loss_curves, *args.layer_loss_curves])
    
    assert len(args.layer_loss_curves) == len(args.hrnn_layer_names), f"Must have same number of hrnn labels as layer loss curves"
    
    with open(args.input, "rb") as input_file:
        training_metrics = pickle.load(input_file)
    
    plot_loss_curves(training_metrics["train_losses"], training_metrics["dev_losses"], args.loss_curves)
    
    plot_layer_losses(training_metrics["train_layer_losses"], training_metrics["dev_layer_losses"], args.hrnn_layer_names, args.layer_loss_curves)
    
    
