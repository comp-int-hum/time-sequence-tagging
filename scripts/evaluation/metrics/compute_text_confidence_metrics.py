import argparse
import json
from utility import make_parent_dirs_for_files
import pickle
from batch_utils import unbatch

def get_confidence_matrix(model_predictions, layer_names, top_k=10, threshold=0.5):
    _, num_layers = model_predictions["true_labels"].shape
    
    layer_prediction_metrics = []
    
    for l, layer_name in enumerate(layer_names):
        # Scores and labels
        layer_scores = model_predictions["scores"][:, l]
        layer_labels = model_predictions["true_labels"][:, l]
        
        # Preds
        predictions = (layer_scores > threshold).long()
        
        # Get indices
        tp_indices = ((predictions == 1) & (layer_labels == 1)).nonzero(as_tuple=True)[0]
        tn_indices = ((predictions == 0) & (layer_labels == 0)).nonzero(as_tuple=True)[0]
        fp_indices = ((predictions == 1) & (layer_labels == 0)).nonzero(as_tuple=True)[0]
        fn_indices = ((predictions == 0) & (layer_labels == 1)).nonzero(as_tuple=True)[0]
        
        # Check min
        num_tp = min(tp_indices.numel(), top_k)
        num_tn = min(tn_indices.numel(), top_k)
        num_fp = min(fp_indices.numel(), top_k)
        num_fn = min(fn_indices.numel(), top_k)
        
        def get_confidence_with_sentence(indices):
            metadata_idx = 0
            entry_dict = []
            for idx in indices:
                if idx >= model_predictions["cumulative_lengths"][metadata_idx]:
                    metadata_idx += 1
                
                entry_dict.append(
                    {
                        "confidence": layer_scores[idx].item(),
                        "sentence": model_predictions["sentences"][idx],
                        "metadata": model_predictions["metadata"][metadata_idx]
                    })
            return sorted(entry_dict, key=lambda x: x["confidence"])
        
        most_confident_tp = get_confidence_with_sentence(tp_indices)[-num_tp:] if tp_indices.numel() > 0 else None
        least_confident_tp = get_confidence_with_sentence(tp_indices)[:num_tp] if tp_indices.numel() > 0 else None
        
        most_confident_tn = get_confidence_with_sentence(tn_indices)[-num_tn:] if tn_indices.numel() > 0 else None
        least_confident_tn = get_confidence_with_sentence(tn_indices)[:num_tn] if tn_indices.numel() > 0 else None
        
        most_confident_fp = get_confidence_with_sentence(fp_indices)[-num_fp:] if fp_indices.numel() > 0 else None
        least_confident_fp = get_confidence_with_sentence(fp_indices)[:num_fp] if fp_indices.numel() > 0 else None
        
        most_confident_fn = get_confidence_with_sentence(fn_indices)[-num_fn:] if fn_indices.numel() > 0 else None
        least_confident_fn = get_confidence_with_sentence(fn_indices)[:num_fn] if fn_indices.numel() > 0 else None
        
        layer_prediction_metrics.append({
            "layer_num": l,
            "layer": layer_name,
            "most_confident_tp": most_confident_tp,
            "least_confident_tp": least_confident_tp,
            "most_confident_tn": most_confident_tn,
            "least_confident_tn": least_confident_tn,
            "most_confident_fp": most_confident_fp,
            "least_confident_fp": least_confident_fp,
            "most_confident_fn": most_confident_fn,
            "least_confident_fn": least_confident_fn,
        })
    
    return layer_prediction_metrics
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Filepath for data containing training outputs")
    parser.add_argument("--confidence_matrix", dest = "confidence_matrix")
    parser.add_argument("--hrnn_layer_names", dest = "hrnn_layer_names", nargs = "+", default = ["paragraphs", "chapters"], help = "Names of hierarchical layers in model")
    parser.add_argument("--threshold", type = float, default = 0.5, help = "Boundary decision threshold")
    args = parser.parse_args()
    
    make_parent_dirs_for_files([args.confidence_matrix])

    with open(args.input, "rb") as input_file:
        batched_model_outputs = pickle.load(input_file)
        seq_len, num_layers_minus_one = batched_model_outputs["guesses"][0][0].shape
    
    assert len(args.hrnn_layer_names) == num_layers_minus_one, "Number of hierarchical layer names provided does not match layers in guesses"
    
    predictions = unbatch(batched_model_outputs)
    
    with open(args.confidence_matrix, "wt") as confidence_matrix_output:
        layerwise_confusion_metrics = get_confidence_matrix(predictions,
                                                           args.hrnn_layer_names,
                                                           top_k = 5,
                                                           threshold = args.threshold)
        
        json.dump(layerwise_confusion_metrics, confidence_matrix_output)
        
    