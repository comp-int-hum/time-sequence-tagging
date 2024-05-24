import argparse
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import h5py
import random

class NarrativeUnitDataset(Dataset):
    def __init__(self, embedding_dir, transform=None, target_transform=None):
        self.embedding_dir = embedding_dir
        with h5py.File(self.embedding_dir, 'r') as hf:
            self.len = len(hf)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.embedding_dir, 'r') as hf:
            group = hf[str(idx)]
            chapters = len(group)
        
        random_chapter_num = random.randrange(chapters-1)

        rand_chapter_embedding = group[str(random_chapter_num)]

        example_type = random.choice("Positive", "Negative")

        if example_type == "Positive":
            # TODO: write positive example code
        else:
            # TODO: write negative example code

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

#
# This script does *nothing* except print out its arguments and touch any files
# specified as outputs (thus fulfilling a build system's requirements for
# success).
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    args, rest = parser.parse_known_args()

    print("Building files {} from arguments {}".format(args.outputs, rest))
    for fname in args.outputs:
        with open(fname, "wt") as ofd:
            pass


def get_batch(filepath, model_type, batch_size = 32, device="cuda"):
    data_batch = []
    label_batch = []
    metadata_batch = []

    curr_data_batch = []
    curr_label_batch = []
    curr_metadata = []
    i = 0
    with gzip.open(filepath, 'r') as gzipfile:
        datapoints = jsonlines.Reader(gzipfile)
        for datapoint in datapoints:
            i+=1
            if not datapoint["segment"]:
                raise ValueError("missing segment")
            curr_data_batch.append(datapoint["segment"])
            curr_label_batch.append(torch.tensor(datapoint["ground_truth"]))

            curr_metadata.append({"id" : datapoint["id"],
                                  "first_ch" : datapoint["first_ch_passage"],
                                  "second_ch" : datapoint["second_ch_passage"],
                                  "ground_truth": datapoint["ground_truth"]})
            
            if len(curr_data_batch) == batch_size:
                data_batch.append(torch.tensor(curr_data_batch).to(device))
                label_batch.append(torch.stack(curr_label_batch).to(device))
                metadata_batch.append(curr_metadata)
                curr_data_batch = []
                curr_label_batch = []
                curr_metadata = []
        if curr_data_batch:
            data_batch.append(torch.tensor(curr_data_batch).to(device))
            label_batch.append(torch.stack(curr_label_batch).to(device))
            metadata_batch.append(curr_metadata)

    print(f"NUM DATAPOINTS IN FILE {i}")
    return (data_batch, label_batch), metadata_batch, i


	def calculate_accuracy(model_type, output, labels, metadata):
    incorrect_texts = []
    if model_type == "sequence_tagger":
        predicted_classes = torch.argmax(output, dim=2)
        correct_predictions = (predicted_classes == labels)
        num_correct = correct_predictions.sum().item()
        total = torch.numel(predicted_classes)
        
        tp_sum, fp_sum, fn_sum = 0, 0, 0 

        for i, correct_item in enumerate(correct_predictions): # (N, L)
            if not correct_item.all():
                error_idxs = (~correct_item).nonzero(as_tuple=False).view(-1).tolist()
                metadata[i]["errors"] = [(labels[i, idx].item(), metadata[i]["original_text"][idx]) for idx in error_idxs]
                TP = (labels[i] & predicted_classes[i]).sum().item()
                FP = ((~labels[i].bool()) & predicted_classes[i]).sum().item()
                FN = (labels[i] & (~predicted_classes[i].bool())).sum().item()
                metadata[i]["accuracy"] = 1 - ((FP + FN)/ len(predicted_classes[i]))
                metadata[i]["precision"] = TP / (TP + FP) if (TP + FP) else float("nan")
                metadata[i]["recall"] = TP / (TP + FN) if (TP + FN) else float("nan")
                tp_sum += TP
                fp_sum += FP
                fn_sum += FN
                incorrect_texts.append(metadata[i])
    else: # classifier
        pred_ind = torch.argmax(output, dim=1) # (N, num_classes)
        correct_predictions = (pred_ind == labels)
        num_correct = (correct_predictions).sum().item()
        
        for i, correct in enumerate(correct_predictions):
            if not correct:
                incorrect_texts.append(metadata[i])
        total = labels.size(0)
    return num_correct, total, tp_sum, fp_sum, fn_sum, incorrect_texts


def evaluate(model, batches, metadata, model_type, device):
    model.eval()
    num_correct = 0
    total = 0
    incorrect_texts = []
    
    input_data, input_labels = batches
    assert len(metadata) == len(input_data)
    assert len(metadata) == len(input_labels)

    TP, FP, FN = 0, 0, 0
    with torch.no_grad():
        for input, label, meta in zip(input_data, input_labels, metadata):
            # If empty batch, pass
            if input.size(0) == 0 or label.size(0) == 0:
                print("Empty batch")
                continue

            input = input.to(device)
            label = label.to(device)

            # Output and loss
            output = model(input)
            correct, curr_total, tp, fp, fn, incorrect = calculate_accuracy(model_type, output, label, meta)
            
			# Get metrics
            num_correct += correct
            total += curr_total
            TP += tp
            FP += fp
            FN += fn
            
			# Add incorrect texts
            incorrect_texts.extend(incorrect)
    
	# Calculate final metrics
    accuracy = num_correct / total
    precision = (TP / (TP + FP)) if (TP + FP) else float("nan")
    recall = (TP / (TP + FN)) if (TP + FN) else float("nan")
    return accuracy, precision, recall, incorrect_texts