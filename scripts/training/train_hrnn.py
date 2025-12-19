import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gzip
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import texttable as tt
from models.hrnn_tagger import HRNN
from models.rnn_tagger import RNNTagger
from models.simple_tagger import SimpleTagger
import json
import pickle
import os
import sys
from utils.batch_utils import get_batch, unpad_predictions
from utils.utility import make_parent_dirs_for_files
from scripts.evaluation.evaluate import apply_metrics
import logging

logger = logging.getLogger(__name__)

def calculate_loss(predictions, teacher_labels, layer_loss_weights, balance_pos_neg = None, device = "cpu"):
    """
    predictions: transition_probs (batch_size, seq_len, num_layers - 1)
    teacher_labels: Tensor of true labels (shape: batch_si[[ze, seq_len, num_layers-1)
    transition_weights: List: (num_layers - 1) - signifies the weights for the transition layers/probs
    
    Returns:
       overall_loss: elementwise loss (averaged over sequence and batch)
       loss_per_layer: elementwise loss for each layer (num_layers - 1)
    """
    logger.debug(f"Calculating loss for predictions shape: {predictions.shape}, labels shape: {teacher_labels.shape}")
    
    # Get tensor shapes/dims
    assert predictions.shape == teacher_labels.shape, "Shape mismatch between predictions and teacher labels"
    batch_size, seq_len, num_layers_minus_one = predictions.shape
    
    # Reshape preds and labels
    predictions = predictions.view(-1, num_layers_minus_one) # (batch_size * seq_len, num_classes)
    teacher_labels = teacher_labels.view(-1, num_layers_minus_one) # (batch_size * seq_len, num_classes)
    
    # Get layer weights and convert to tensor
    layer_weights_tensor = torch.tensor(layer_loss_weights, dtype=predictions.dtype, device = device)
    
    # Define loss function
    loss_fn = nn.BCELoss(reduction = "none")
    
    # Get loss per element
    ele_loss = loss_fn(predictions, teacher_labels.float()) # (batch_size * seq_len, num_classes)
    
    # Weight loss with layer_weights --> broadcasting across last dim = num_classes: (batch_size * seq_len, num_classes)
    weighted_loss = ele_loss * layer_weights_tensor
    
    # Positive / negative balancing
    if balance_pos_neg:
        assert len(balance_pos_neg) == 2
        
        pos_weight = torch.tensor(balance_pos_neg[0], dtype = predictions.dtype, device = device)
        neg_weight = torch.tensor(balance_pos_neg[1], dtype = predictions.dtype, device = device)
        
        # Get masks
        pos_mask = (teacher_labels == 1).float()
        neg_mask = (teacher_labels == 0).float()
        
        # Pos and neg losses per layer
        pos_loss_per_layer = (pos_mask * weighted_loss).sum(dim=0) / (pos_mask.sum(dim=0) + 1e-6) * pos_weight # (num_classes)
        neg_loss_per_layer = (neg_mask * weighted_loss).sum(dim=0) / (neg_mask.sum(dim=0) + 1e-6) * neg_weight # (num_classes)
        
        # TODO: fix this --> the balancing doesn't make sense
        overall_loss = (pos_loss_per_layer.mean() + neg_loss_per_layer.mean()) / 2
        loss_per_layer = (pos_loss_per_layer + neg_loss_per_layer) / 2  # Loss for each layer
        
        logger.debug(f"Balanced loss - positive examples: {pos_mask.sum().item()}, negative examples: {neg_mask.sum().item()}")
    else:
        overall_loss = weighted_loss.mean()
        
        # Reduce dim = 0 (aka non-num_classes)
        loss_per_layer = weighted_loss.mean(dim=0)

    logger.debug(f"Overall loss: {overall_loss.item():.6f}, Per-layer losses: {loss_per_layer.tolist()}")
    return overall_loss, loss_per_layer


def run_model(model, optimizer, batches, layer_weights, balance_pos_neg, device="cpu", is_train=True, teacher_ratio = 0.6, temperature = 1.0):
    """Evaluate model given batches, metadata, and class labels"""
    mode = "Training" if is_train else "Evaluation"
    logger.info(f"{mode} mode - processing {len(batches[0])} batches")
    
    if is_train:
        model.train()
    else:
        model.eval()
    
    guesses = []
    golds = []
    layer_losses = []
    running_loss = 0
    
    with torch.set_grad_enabled(is_train):
        for batch_idx, (input, labels) in enumerate(tqdm(zip(*batches), desc = f"{mode} loop")):

            if is_train:
                optimizer.zero_grad()
                
            # If empty batch, pass
            if input.size(0) == 0:
                logger.warning(f"Empty batch encountered at index {batch_idx}")
                continue
            
            # Move to device
            input = input.to(device)
            
            logger.debug(f"Batch {batch_idx}: input shape {input.shape}, labels shape {labels.shape}")
            
            out = model(input, teacher_forcing = labels, teacher_ratio = teacher_ratio, temperature = temperature)
            loss, loss_per_layer = calculate_loss(out, labels, layer_weights, balance_pos_neg, device)
            
            golds.append(labels.detach().cpu())
            guesses.append(out.detach().cpu())
            layer_losses.append(loss_per_layer.detach().cpu())
            
            if is_train:                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            # Log every 10
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(batches[0])}: loss = {loss.item():.6f}")
    
    avg_loss = running_loss / len(batches[0])
    avg_layer_losses = torch.stack(layer_losses).mean(dim=0).tolist()
    
    logger.info(f"{mode} completed - Average loss: {avg_loss:.6f}")
    logger.info(f"Average layer losses: {avg_layer_losses}")
    
    return avg_loss, avg_layer_losses, (guesses, golds)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Hierarchical RNN for sequence boundary detection")
    parser.add_argument("--train", dest="train", nargs = 3, required=True, help="Train data, labels, and embedding file in that order")
    parser.add_argument("--dev", dest="dev", nargs = 3, required=True, help="Dev data, labels, and embedding file in that order")
    parser.add_argument("--architecture", dest = "architecture", help = "Architecture type (HRNN, RNN, NaiveClassifier)")
    
    parser.add_argument("--train_output", dest="train_output", required=True, help="Train data output")
    parser.add_argument("--dev_output", dest="dev_output", required=True, help="Dev data output")
    parser.add_argument("--train_stats", dest="train_stats", required=True, help="Training statistics summary file")
    parser.add_argument("--train_dump", dest="train_dump", required=True, help="All training data dumped (loss, metrics)")
    parser.add_argument("--trained_model", dest="trained_model", required=True, help="Trained model")
    
    
    parser.add_argument("--teacher_ratio", type=float, help="Teacher ratio to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold for training loop")
    parser.add_argument("--hrnn_layer_names", dest="hrnn_layer_names", nargs="+", default=["sentences", "paragraphs", "chapters"], help="Names of hierarchical layers in model")
    parser.add_argument("--temperature", type=float, dest="temperature", default=1.0, help="Temperature for inference")
    
    # Training params
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, required=True, help="number of epochs to train")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.4)
    parser.add_argument("--layer_loss_weights", dest="layer_loss_weights", nargs="*", type=float, help="Layer loss weights for transition layers")
    parser.add_argument("--balance_pos_neg", nargs=2, type=float, help="Whether to balance positive and negative examples")
    
    args, rest = parser.parse_known_args()
    
    logger.info(
        f"Starting HRNN training with config:\n"
        f"  Train file: {args.train}\n"
        f"  Dev file: {args.dev}\n"
        f"  Epochs: {args.num_epochs} | Batch size: {args.batch_size}\n"
        f"  Dropout: {args.dropout} | Teacher ratio: {args.teacher_ratio}\n"
        f"  Temperature: {args.temperature} | Threshold: {args.threshold}\n"
        f"  Model layers: {args.hrnn_layer_names} | Balance pos/neg: {args.balance_pos_neg}"
    )

    make_parent_dirs_for_files([args.train_dump, args.train_stats, args.train_output, args.dev_output, args.trained_model])
    
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA IN USE")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        logger.info("cpu")
    
    # Get batches
    train_batches = get_batch(*args.train, batch_size=args.batch_size, device=device)
    dev_batches = get_batch(*args.dev, batch_size=args.batch_size, device=device)
    logger.info(f"Loaded {len(train_batches['inputs'][0])} train batches")
    logger.info(f"Loaded {len(dev_batches['inputs'][0])} dev batches")
    
    metrics = [f1_score, recall_score, precision_score, accuracy_score]
    
    # Get data sizes
    try:
        with gzip.open(args.train[2], "rt") as efd:
            j = json.loads(efd.readline())
            emb_dim = len(j["embeddings"][0])
            logger.info(f"Embedding dimension: {emb_dim}")

        with gzip.open(args.train[1], "rt") as lfd:
            l = json.loads(lfd.readline())
            num_transitions = len(l["labels"].keys())
            logger.info(f"Number of transitions: {num_transitions}")

    except Exception as e:
        logger.error(f"Error getting data dimensions: {str(e)}")
        raise

    layer_loss_weights = args.layer_loss_weights if args.layer_loss_weights else [1.0] * num_transitions
    logger.info(f"Layer loss weights: {layer_loss_weights}")

    match args.architecture:
        case "HRNN":
            model = HRNN(
                input_size = emb_dim,
                hidden_size = 512,
                num_layers = 3,
                layer_names = args.hrnn_layer_names,
                dropout = args.dropout,
                device = device
            )
        case "RNN":
            model = RNNTagger(
                lstm_input_size = emb_dim,
                task_num=2,                 # number of classification heads (e.g., for paragraphs and chapters)
                # output_layers=1,            
                # lstm_num_layers=2,        
                # lstm_hidden_size=256,
                # mlp_layer_sizes=[128, 64],
                dropout=args.dropout
            )
        case "SimpleTagger":
            model = SimpleTagger(
                input_size = emb_dim,
                task_num = 2,
                task_size = 1,
            )
        case _:
            raise ValueError(f"No matching tagger for name: {args.architecture}")
    
    logger.info(f"Model architecture:\n{model}")
    
    # Total params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Losses & layer_losses
    train_losses = []
    dev_losses = []
    
    train_lls = []
    dev_lls = []
    
    # Best model
    best_model = None
    best_model_epoch = 0
    best_dev_loss = float("inf")
    to_save_train = None
    to_save_dev = None
    
    # Metrics
    epoch_metrics = []
    
    # Set up summary table
    summary_table = tt.Texttable()
    summary_table.set_cols_width([10, 10, 40, 10, 40])
    summary_table.set_cols_align(["l", "l", "l", "l", "l"])
    summary_table.header(["Epoch", "Train Loss", "Train Layer Losses", "Dev loss", "Dev Layer Losses"])
    
    logger.info("Starting training loop...")
    
    # Training Loop
    try:
        for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
            logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")

            train_loss, train_layer_losses, (train_guesses, train_golds) = run_model(
                model,
                optimizer,
                train_batches["inputs"],
                layer_loss_weights,
                args.balance_pos_neg,
                device=device,
                is_train=True,
                teacher_ratio=args.teacher_ratio
            )
            
            dev_loss, dev_layer_losses, (dev_guesses, dev_golds) = run_model(
                model,
                None,
                dev_batches["inputs"],
                layer_loss_weights,
                args.balance_pos_neg,
                device=device,
                is_train=False,
                teacher_ratio=0.0
            )
            
            train_losses.append(train_loss)
            train_lls.append(train_layer_losses)
            
            dev_losses.append(dev_loss)
            dev_lls.append(dev_layer_losses)
            
            # Calculate metrics
            predictions = {
                "guesses": dev_guesses,
                "golds": dev_golds,
                "lengths": dev_batches["lengths"]
            }
            
            unpadded_dev_outputs = unpad_predictions(predictions)
            
            epoch_metrics.append(apply_metrics(
                unpadded_dev_outputs["scores"],
                unpadded_dev_outputs["true_labels"],
                metrics,
                args.hrnn_layer_names[1:],
                args.threshold
            ))
            
            # Check for best model
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_model_epoch = epoch
                best_model = model.state_dict()
                to_save_train = train_guesses
                to_save_dev = dev_guesses
                logger.info(f"New best model found at epoch {epoch} with dev loss: {dev_loss:.6f}")
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1} completed:\n"
                f"  Train loss: {train_loss:.6f}\n"
                f"  Dev loss: {dev_loss:.6f}\n"
                f"  Train layer losses: {train_layer_losses}\n"
                f"  Dev layer losses: {dev_layer_losses}"
            )
            
            summary_table.add_row([epoch, f"{train_loss:.6f}", train_layer_losses, f"{dev_loss:.6f}", dev_layer_losses])
        
        logger.info(f"Best model was found at epoch {best_model_epoch + 1} with dev loss: {best_dev_loss:.6f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    
    try:
        with open(args.train_stats, "w") as summary_file:
            summary_file.write(summary_table.draw())
        
        if best_model is not None:
            torch.save(best_model, args.trained_model)
        
        # Save train outputs
        with open(args.train_output, "wb") as train_output_file:
            pickle.dump(
                {
                    "final_guesses": train_guesses,
                    "guesses": to_save_train,
                    "golds": train_golds,
                    "metadata": train_batches["metadata"],
                    "lengths": train_batches["lengths"],
                    "sentences": train_batches["sentences"],
                    "best_model_epoch": best_model_epoch
                },
                train_output_file
            )
            
        # Save dev outputs
        with open(args.dev_output, "wb") as dev_output_file:
            pickle.dump(
                {
                    "final_guesses": dev_guesses,
                    "guesses": to_save_dev,
                    "golds": dev_golds,
                    "metadata": dev_batches["metadata"],
                    "lengths": dev_batches["lengths"],
                    "sentences": dev_batches["sentences"],
                    "best_model_epoch": best_model_epoch
                },
                   dev_output_file
            )
        logger.info(f"Development outputs saved to: {args.dev_output}")
            
        # Save training statistics
        with open(args.train_dump, "wb") as training_stats_file:
            pickle.dump(
                {
                    "train_losses": train_losses,
                    "train_layer_losses": train_lls,
                    "dev_losses": dev_losses,
                    "dev_layer_losses": dev_lls,
                    "best_model_epoch": best_model_epoch,
                    "epoch_metrics": epoch_metrics
                },
                   training_stats_file
            )
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise
