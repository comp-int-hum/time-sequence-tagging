import jsonlines
from utility import open_file
import torch.nn.utils
import torch
# from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils
from itertools import accumulate

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
    data_batches, label_batches, metadata_batches, length_batches, sentence_batches = [], [], [], [], []
    data_batch, label_batch, metadata_batch, sentence_batch = [], [], [], []
    
    # Helper function for appending batches and padding if model_type is sequence_tagger
    def append_data(batch_data, batch_label, batch_metadata, batch_sentence):
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
        sentence_batches.append(batch_sentence)
        
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

def unpad_predictions(predictions):
    # Unpad guesses and gold labels
    unpadded_guesses = []
    unpadded_golds = []
    
    for guess, gold, true_length in zip(predictions["guesses"], predictions["golds"], predictions["lengths"]):
        unpadded_guesses.extend(rnn_utils.unpad_sequence(guess, torch.Tensor(true_length), batch_first=True))
        unpadded_golds.extend(rnn_utils.unpad_sequence(gold, torch.Tensor(true_length), batch_first=True))

    scores = torch.cat(unpadded_guesses, dim=0)  # Shape: (-1, num_layers)
    true_labels = torch.cat(unpadded_golds, dim=0)  # Shape: (-1, num_layers)
    
    return {
		"scores": scores, # sentence-level (batch_size * seq_len, num_layers)
        "true_labels": true_labels, # sentence-level (batch_size * seq_len, num_layers)
	}

def unbatch(batch_predictions):
    
    return {
        **unpad_predictions(batch_predictions),
        "sentences": [sentence for batch in batch_predictions["sentences"] for sublist in batch for sentence in sublist],
        "metadata": [s for batch in batch_predictions["metadata"] for s in batch],
        "true_lengths": (true_lengths:=[l for batch in batch_predictions["lengths"] for l in batch]),
        "cumulative_lengths": list(accumulate(true_lengths))
        # "guesses": unpadded_guesses, # list of length batch_size [(seq_len, num_layers)]
        # "golds": unpadded_golds,  # list of length batch_size [(seq_len, num_layers)]
    }