import argparse
import json
import os
from utility import make_dirs
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import torch
import re
import numpy as np
import pickle
import torch.nn.utils.rnn as rnn_utils
from itertools import accumulate

def sanitize_filename(title, max_length=255):
    sanitized = re.sub(r'[\/:*?"<>|]', '_', title)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized[:max_length]

def build_visualization_output_name(root_dir, title, author):
    title_author = sanitize_filename(f"{title}-{author}")
    vis_path_name = f"{root_dir}/{title_author}.png"
    os.makedirs(os.path.dirname(vis_path_name), exist_ok=True)
    return vis_path_name

def get_general_roc_curve(guesses, golds, true_lengths, save_path):
    """_summary_

    Args:
        guesses (List): (seq_len, num_layers - 1)
        golds (List): (seq_len, num_layers - 1)
        true_lengths (List[Int]): true_length for each
        save_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    unpadded_guesses = []
    unpadded_golds = []
    for guess, gold, true_length in zip(guesses, golds, true_lengths):
        unpadded_guesses.append(guess[:, :true_length, :].flatten())
        unpadded_golds.append(gold[:, :true_length, :].flatten())
        
    scores = torch.cat(unpadded_guesses, dim = 0).numpy()
    true_labels = torch.cat(unpadded_golds, dim = 0).numpy()
    
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name='General estimator')
    display.plot()
    plt.show()
    plt.savefig(save_path)
    return (fpr, tpr, thresholds)

def get_separated_roc_curves(guesses, golds, true_lengths, save_paths):
    """_summary_

    Args:
        guesses (List): (batch_size, seq_len, num_layers - 1)
        golds (List): (batch_size, seq_len, num_layers - 1)
        true_lengths (List[Int]): true_length for each
        save_paths (_type_): _description_

    Returns:
        _type_: _description_
    """
    batch_size, seq_len, num_layers = guesses[0].shape
    
    unpadded_guesses = []
    unpadded_golds = []
    for guess, gold, true_length in zip(guesses, golds, true_lengths):
        unpadded_guesses.extend(rnn_utils.unpad_sequence(guess, torch.Tensor(true_length), batch_first = True))
        unpadded_golds.extend(rnn_utils.unpad_sequence(gold, torch.Tensor(true_length), batch_first = True))
        # guess[:, :true_length, :].view(-1, num_layers))
        # unpadded_golds.append(gold[:, :true_length, :].view(-1, num_layers))
        
    scores = torch.cat(guesses, dim = 0)
    true_labels = torch.cat(golds, dim = 0)
    
    layer_stats = []
    # (f"NUM LAYERS IN SEPARATED: {num_layers}")
    for l in range(num_layers):
        fpr, tpr, thresholds = roc_curve(true_labels[:, :, l].numpy(), scores[:, :, l].numpy(), pos_label=1)
        roc_auc = auc(fpr, tpr)
        layer_stats.append((fpr, tpr, thresholds))
        
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name=f'Layer {l} estimator')
        display.plot()
        plt.show()
        plt.savefig(save_paths[l])
        
    return layer_stats

def plot_and_save_roc(true_labels, scores, estimator_name, save_path):
    fpr, tpr, thresholds = roc_curve(true_labels, scores, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr,
                              tpr=tpr,
                              roc_auc=roc_auc,
                              estimator_name = estimator_name)
    display.plot()
    plt.show()
    plt.savefig(save_path)
    return (fpr, tpr, thresholds)

def get_roc_curves(guesses, golds, true_lengths, save_paths):
    batch_size, seq_len, num_layers = guesses[0].shape
    unpadded_guesses = []
    unpadded_golds = []
    for guess, gold, true_length in zip(guesses, golds, true_lengths):
        unpadded_guesses.extend(rnn_utils.unpad_sequence(guess, torch.Tensor(true_length), batch_first = True))
        unpadded_golds.extend(rnn_utils.unpad_sequence(gold, torch.Tensor(true_length), batch_first = True))
        
    scores = torch.cat(unpadded_guesses, dim = 0) # (-1, num_layers - 1)
    true_labels = torch.cat(unpadded_golds, dim = 0) # (-1, num_layers - 1)
    
    assert scores.shape == true_labels.shape
    
    general_roc_stats = plot_and_save_roc(true_labels.flatten().numpy(), scores.flatten().numpy(), "General Estimator", save_paths[0])
    
    layer_stats = []
    for l in range(num_layers):
        layer_roc_stats = plot_and_save_roc(true_labels[:, l].numpy(), scores[:, l].numpy(), f"Layer {l} estimator", save_paths[l + 1])
        layer_stats.append(layer_roc_stats)
        
    return (general_roc_stats, layer_stats)


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
    plt.title("Boundary Visualization")
    plt.xlabel("num_layers - 1")
    plt.ylabel("Sequence Length (seq_len)")
    
    tick_labels = [f"Pred{i//2}" if i % 2 == 0 else f"Gold{i//2}" for i in range(2 * num_layers)]
    plt.yticks(range(2 * num_layers), tick_labels)
    plt.xticks(range(seq_len))
    
    plt.show()
    plt.savefig(save_path)

def get_confusion_matrix(guesses, golds, true_lengths, sentences, metadata, top_k = 10, threshold = 0.5):
    batch_size, seq_len, num_layers = guesses[0].shape

    # Unpad guesses and gold labels
    unpadded_guesses = []
    unpadded_golds = []
    
    flattened_true_lengths = [t for batch in true_lengths for t in batch]
    cumulative_lengths = list(accumulate(flattened_true_lengths))
        
    for guess, gold, true_length in zip(guesses, golds, true_lengths):
        unpadded_guesses.extend(rnn_utils.unpad_sequence(guess, torch.Tensor(true_length), batch_first=True))
        unpadded_golds.extend(rnn_utils.unpad_sequence(gold, torch.Tensor(true_length), batch_first=True))

    scores = torch.cat(unpadded_guesses, dim=0)  # Shape: (-1, num_layers)
    true_labels = torch.cat(unpadded_golds, dim=0)  # Shape: (-1, num_layers)

    flattened_sentences = [sentence for batch in sentences for sublist in batch for sentence in sublist]  # Flatten sentences
    flattened_metadata = [d for batch in metadata for d in batch]
    layer_stats = []

    for l in range(num_layers):
        layer_scores = scores[:, l]
        layer_labels = true_labels[:, l]

        # probabilities = layer_scores  # torch.sigmoid(layer_scores)
        # Do we want to use raw logits instead?

        predictions = (layer_scores > threshold).long()
        correct_indices = (predictions == layer_labels).nonzero(as_tuple=True)[0] # should be in sorted order
        incorrect_indices = (predictions != layer_labels).nonzero(as_tuple=True)[0]
        
        num_correct = min(correct_indices.numel(), top_k)
        num_incorrect = min(incorrect_indices.numel(), top_k)
        
        def get_confidence_with_sentence(indices):
            metadata_idx = 0
            entry_dict = []
            for idx in indices:
                if idx >= cumulative_lengths[metadata_idx]:
                    metadata_idx += 1
                
                entry_dict.append(
                    {
                        "confidence": layer_scores[idx].item(),
                        "sentence": flattened_sentences[idx],
                        "metadata": flattened_metadata[metadata_idx]
                    })
            return sorted(entry_dict, key=lambda x: x["confidence"])

        most_confident_correct = get_confidence_with_sentence(correct_indices)[-num_correct:] if correct_indices.numel() > 0 else None
        least_confident_correct = get_confidence_with_sentence(correct_indices)[:num_correct] if correct_indices.numel() > 0 else None

        most_confident_incorrect = get_confidence_with_sentence(incorrect_indices)[-num_incorrect:] if incorrect_indices.numel() > 0 else None
        least_confident_incorrect = get_confidence_with_sentence(incorrect_indices)[:num_incorrect] if incorrect_indices.numel() > 0 else None
        
        print(f"Most confident: {most_confident_correct}")
        layer_stats.append({
            "layer": l,
            "most_confident_correct": most_confident_correct,
            "least_confident_correct": least_confident_correct,
            "most_confident_incorrect": most_confident_incorrect,
            "least_confident_incorrect": least_confident_incorrect,
        })

    return layer_stats

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Filepath for data containing training outputs")
    parser.add_argument("--layer_names", dest = "layer_names", nargs = "+", default = ["paragraphs", "chapters"], help = "Names of hierarchical layers in model")
    parser.add_argument("--visualization", dest = "visualization", help = "Visualization output path")
    parser.add_argument("--confusion_matrix", dest = "confusion_matrix")
    parser.add_argument("--vis_num", type = int, help = "Number of samples to visualize")
    parser.add_argument("--threshold", type = float, default = 0.5, help = "Boundary decision threshold")
    args = parser.parse_args()
    
    with open(args.input, "rb") as input_file:
        train_outputs = pickle.load(input_file)
        seq_len, num_layers_minus_one = train_outputs["train_guesses"][0][0].shape
    
    print(f"Seq len: {seq_len}")
    print(f"Num layers minus one: {num_layers_minus_one}")
    os.makedirs(args.visualization, exist_ok=True)
    
    assert len(args.layer_names) == num_layers_minus_one, "Number of hierarchical layer names provided does not match layers in guesses"
    
    roc_save_paths = [f"{args.visualization}/general_roc_curve.png"] + [f"{args.visualization}/roc_curve_for_{args.layer_names[l]}.png" for l in range(num_layers_minus_one)]
    
    (dev_roc_stats, layer_stats) = get_roc_curves(train_outputs["dev_guesses"],
                                                  train_outputs["dev_golds"],
                                                  train_outputs["dev_lengths"],
                                                  roc_save_paths)
    
    with open(args.confusion_matrix, "wt") as confusion_matrix_output:
        layer_stats = get_confusion_matrix(train_outputs["dev_guesses"],
                                           train_outputs["dev_golds"],
                                           train_outputs["dev_lengths"],
                                           train_outputs["dev_sentences"],
                                           train_outputs["dev_metadata"],
                                           top_k = 5,
                                           threshold = args.threshold)
        json.dump(layer_stats, confusion_matrix_output)
        
    for i, (dev_guess, dev_gold, dev_meta, dev_lengths) in enumerate(list(zip(train_outputs["dev_guesses"],
                                                                              train_outputs["dev_golds"],
                                                                              train_outputs["dev_metadata"],
                                                                              train_outputs["dev_lengths"]))[:args.vis_num]):
        output_name = build_visualization_output_name(args.visualization, dev_meta[0]['title'], dev_meta[0]['author'])
        create_boundary_visualization(dev_guess[0], dev_gold[0], dev_lengths[0], save_path = output_name)