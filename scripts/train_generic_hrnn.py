import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utility import make_parent_dirs_for_files
import logging
from tqdm import tqdm
import gzip
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score # , ConfusionMatrixDisplay, confusion_matrix,
import texttable as tt
from generic_hrnn import GenericHRNN
import texttable as tt
import json
import pickle
import os
import torch
from batch_utils import get_batch, unpad_predictions
from evaluate import get_f1_score, apply_metrics

logger = logging.getLogger("train_sequence_model")

def calculate_loss(predictions, teacher_labels, layer_weights, balance_pos_neg = None, device = "cpu"):
    """
    predictions: transition_probs (batch_size, seq_len, num_layers - 1)
    teacher_labels: Tensor of true labels (shape: batch_si[[ze, seq_len, num_layers-1)
    layer_weights: List: (num_layers - 1)
    
    Returns:
       overall_loss: elementwise loss (averaged over sequence and batch)
       loss_per_layer: elementwise loss for each layer (num_layers - 1)
    """
    # Get tensor shapes/dims
    assert predictions.shape == teacher_labels.shape, "Shape mismatch between predictions and teacher labels"
    batch_size, seq_len, num_layers_minus_one = predictions.shape
    
    # Reshape preds and labels
    predictions = predictions.view(-1, num_layers_minus_one) # (batch_size * seq_len, num_classes)
    teacher_labels = teacher_labels.view(-1, num_layers_minus_one) # (batch_size * seq_len, num_classes)
    
    # Get layer weights and convert to tensor
    layer_weights_tensor = torch.tensor(layer_weights, dtype=predictions.dtype, device = device)
    
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
    else:
        overall_loss = weighted_loss.mean()
        
        # Reduce dim = 0 (aka non-num_classes)
        loss_per_layer = weighted_loss.mean(dim=0)

    return overall_loss, loss_per_layer


def run_model(model, optimizer, batches, layer_weights, balance_pos_neg, device="cpu", is_train=True, teacher_ratio = 0.6, temperature = 1.0):
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
            
            out = model(input, teacher_forcing = labels, teacher_ratio = teacher_ratio, temperature = temperature)
            loss, loss_per_layer = calculate_loss(out, labels, layer_weights, balance_pos_neg, device)
            
            golds.append(labels.detach().cpu())
            guesses.append(out.detach().cpu())
            layer_losses.append(loss_per_layer.detach().cpu())
            
            if is_train:                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
    
    return (running_loss / len(batches)), torch.stack(layer_losses).mean(dim=0).tolist(), (guesses, golds)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest = "train", help = "Train file")
    parser.add_argument("--dev", dest = "dev", help = "Dev file")
    
    parser.add_argument("--train_output", dest = "train_output", help = "Train data output")
    parser.add_argument("--dev_output", dest = "dev_output", help = "Dev data output")
    parser.add_argument("--training_summary", dest = "training_summary", help = "Training summary results file")
    parser.add_argument("--training_stats", dest = "training_stats", help = "Training statistics (loss, metrics)")
    parser.add_argument("--model", dest="model", help = "Trained model")
    
    
    parser.add_argument("--teacher_ratio", type = float, help = "Teacher ratio to use")
    parser.add_argument("--threshold", type = float, default = 0.5, help = "Prediction threshold for training loop")
    parser.add_argument("--hrnn_layer_names", dest = "hrnn_layer_names", nargs = "+", default = ["paragraphs", "chapters"], help = "Names of hierarchical layers in model")
    parser.add_argument("--temperature", type = float, dest = "temperature", default = 1.0, help = "Temperature for inference")
    
    # Training params
    parser.add_argument("--num_epochs", dest="num_epochs", type = int, help="number of epochs to train")
    parser.add_argument("--batch_size", dest="batch_size", type = int, default=32, help="Batch size")
    parser.add_argument("--dropout", dest="dropout", type=float, default = 0.4)
    parser.add_argument("--layer_weights", dest = "layer_weights", nargs = "*", type = float, help = "Weights for different hierarchical layers") # required = False, 
    parser.add_argument("--balance_pos_neg", nargs = 2, type = float, help = "Whether to balance positive and negative examples") # required = False, action = "store_true",
    
    args, rest = parser.parse_known_args()
    
    make_parent_dirs_for_files([args.training_summary, args.train_output, args.dev_output, args.model])
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Get batches
    train_batches = get_batch(args.train, batch_size=args.batch_size, device = device)
    dev_batches = get_batch(args.dev, batch_size=args.batch_size, device = device)
    
    metrics = [f1_score, recall_score, precision_score, accuracy_score]
    
    # Get data sizes
    with gzip.open(args.train, "rt") as ifd:
        j = json.loads(ifd.readline())
        emb_dim = len(j["flattened_embeddings"][0])
        num_layers_minus_one = len(j["hierarchical_labels"][0])
        print(f"Num layers - 1: {num_layers_minus_one}")
    
    layer_weights = args.layer_weights if args.layer_weights else [1.0] * num_layers_minus_one
    print(f"Layer weights: {layer_weights}")
    
    model = GenericHRNN(
        input_size = emb_dim,
        hidden_size = 512,
        num_layers = 3,
        layer_names=args.hrnn_layer_names,
        dropout = 0.,
        device = device
    )
    
    logger.info("%s", model)

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
    
    # Training Loop
    for epoch in tqdm(range(args.num_epochs), desc = "Epochs"):

        train_loss, train_layer_losses, (train_guesses, train_golds) = run_model(model,
                                                                                 optimizer,
                                                                                 train_batches["inputs"],
                                                                                 layer_weights,
                                                                                 args.balance_pos_neg,
                                                                                 device = device,
                                                                                 is_train=True,
                                                                                 teacher_ratio = args.teacher_ratio)
        
        dev_loss, dev_layer_losses, (dev_guesses, dev_golds) = run_model(model,
                                                                         None,
                                                                         dev_batches["inputs"],
                                                                         layer_weights,
                                                                         args.balance_pos_neg,
                                                                         device = device,
                                                                         is_train = False,
                                                                         teacher_ratio = 0.0)
        
        train_losses.append(train_loss)
        train_lls.append(train_layer_losses)
        
        dev_losses.append(dev_loss)
        dev_lls.append(dev_layer_losses)
        
        predictions = {
            "guesses": dev_guesses,
            "golds": dev_golds,
            "lengths": dev_batches["lengths"]
        }
        
        unpadded_dev_outputs = unpad_predictions(predictions)
        
        epoch_metrics.append(apply_metrics(unpadded_dev_outputs["scores"],
                                           unpadded_dev_outputs["true_labels"],
                                           metrics,
                                           args.hrnn_layer_names,
                                           args.threshold))
        
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model_epoch = epoch
            best_model = model.state_dict()
            to_save_train = train_guesses
            to_save_dev = dev_guesses
            
        summary_table.add_row([epoch, train_loss, train_layer_losses, dev_loss, dev_layer_losses])
    
    with open(args.training_summary, "w") as summary_file:
        summary_file.write(summary_table.draw())
    
    print(f"Size of dev guesses {len(dev_guesses)}, dev_golds {len(dev_golds)}, metadata: {len(dev_batches['metadata'])}")
    
    if best_model is not None:
        torch.save(best_model, args.model)
    
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
        
    with open(args.training_stats, "wb") as training_stats_file:
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
