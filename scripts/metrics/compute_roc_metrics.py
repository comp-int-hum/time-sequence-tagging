import argparse
import json
import os
from utility import make_parent_dirs_for_files
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import matthews_corrcoef
import numpy as np
import pickle
from batch_utils import unbatch
from sklearn.metrics import f1_score
import gzip

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

def compute_layerwise_roc(model_predictions, save_paths):
    _ , num_layers = model_predictions["true_labels"].shape
    
    assert model_predictions["scores"].shape == model_predictions["true_labels"].shape
    
    layerwise_roc_metrics = []
    layerwise_optimal_thresholds = []
    for l in range(num_layers):
        layer_labels = model_predictions["true_labels"][:, l].numpy()
        layer_scores =  model_predictions["scores"][:, l].numpy()
        roc_metrics = plot_and_save_roc(layer_labels,
                                        layer_scores,
                                        f"Layer {l} estimator",
                                        save_paths[l])
        layerwise_roc_metrics.append(roc_metrics)
        layerwise_optimal_thresholds.append(get_optimal_thresholds(*roc_metrics, layer_labels, layer_scores))
      
    return {
        "roc_metrics": layerwise_roc_metrics,
        "optimal_thresholds": layerwise_optimal_thresholds
    }

def get_optimal_thresholds(fpr, tpr, thresholds, labels, scores):
    j_scores = tpr - fpr
    j_optimal = thresholds[np.argmax(j_scores)]
    
    f1_scores = [f1_score(labels, scores >= thr) for thr in thresholds]
    f1_optimal = thresholds[np.argmax(f1_scores)]
    
    mcc_scores = [matthews_corrcoef(labels, scores >= thr) for thr in thresholds]
    mcc_optimal = thresholds[np.argmax(mcc_scores)]
    
    return {
        "youden_j": float(j_optimal),
        "f1": float(f1_optimal),
        "mcc": float(mcc_optimal)
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Filepath for data containing training outputs")
    parser.add_argument("--threshold_metrics", dest = "threshold_metrics", help = "Output file for different optimal thresholds")
    parser.add_argument("--roc_by_layer", dest = "roc_by_layer", nargs = "+", help = "Output roc by layer")
    parser.add_argument("--hrnn_layer_names", dest = "hrnn_layer_names", nargs = "+", default = ["paragraphs", "chapters"], help = "Names of hierarchical layers in model")
    args = parser.parse_args()
    
    make_parent_dirs_for_files([*args.roc_by_layer, args.threshold_metrics])
    print(f"ROC by layer: {args.roc_by_layer}")
    
    with open(args.input, "rb") as input_file:
        batched_model_outputs = pickle.load(input_file)
        seq_len, num_layers_minus_one = batched_model_outputs["guesses"][0][0].shape
        
    assert len(args.hrnn_layer_names) == num_layers_minus_one, "Number of hierarchical layer names provided does not match layers in guesses"
    
    predictions = unbatch(batched_model_outputs)
    
    print(type(predictions["cumulative_lengths"]))
    
    layerwise_rocs = compute_layerwise_roc(predictions,
                                           args.roc_by_layer)
    
    with open(args.threshold_metrics, "w") as threshold_output:
         json.dump(dict(zip(args.hrnn_layer_names, layerwise_rocs["optimal_thresholds"])),
                   threshold_output)
    
    
    
        
    