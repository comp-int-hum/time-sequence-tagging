import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import json
import os
from utility import make_dirs
import gzip

class BasicClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        print(f"Input size: {input_size}")
        self.fc1 = nn.Linear(input_size, 256) # batch_first, H_in
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in")
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in")
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        nn.init.kaiming_normal_(self.fc3.weight, mode="fan_in")
        self.output = nn.Linear(256, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        # print(f"First: {first.shape}")
        # print(f"Second: {second.shape}")
        # x = torch.cat((first, second), dim = 0).to("cuda")
        # print(f"Cat {x.shape}")
        x = self.fc1(data)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.output(x)
        # x = self.softmax(x)
        return x

def get_batch(filepath, batch_size = 32, class_size = 2, device="cuda"):
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
            # one_hot = nn.functional.one_hot(torch.tensor(datapoint["ground_truth"]), num_classes=class_size)
            # curr_label_batch.append(one_hot)
            curr_metadata.append({"id" : datapoint["id"],
                                  "chapters" : datapoint["chapter_names"],
                                  "first_ch" : datapoint["first_ch_idxs"],
                                  "second_ch" : datapoint["second_ch_idxs"]})
            
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

    print(f"NUM DTAPOINTS IN FILE {i}")
    return (data_batch, label_batch), metadata_batch

def evaluate(model, batches, metadata, device):
    model.eval()
    num_correct = 0
    total = 0
    incorrect_texts = []
    
    input_data, input_labels = batches
    with torch.no_grad():
        for input, labels, meta in zip(input_data, input_labels, metadata):
            if input.size(0) == 0 or labels.size(0) == 0:
                print("Empty batch")
                continue

            input = input.to(device)
            labels = labels.to(device)

            # Output and loss
            output = model(input).squeeze(dim=1)
            pred_ind = torch.argmax(output, dim=1)
            # truth = torch.argmax(labels, dim=1)
            # correct_preds = (pred_ind == truth)
            correct_preds = (pred_ind == labels)
            num_correct += (correct_preds).sum().item()

            for i, correct in enumerate(correct_preds):
                if not correct:
                    incorrect_texts.append(meta[i])

            # correct += (torch.abs(labels - output) < 0.5).sum().item()
            total += labels.size(0)
    accuracy = num_correct / total
    return accuracy, incorrect_texts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", help="name of training datapoints file")
    parser.add_argument("--test", dest="test", help="name of test datapoints file")
    parser.add_argument("--model_name", dest="model_name", help="Name of best model")
    parser.add_argument("--emb_dim", dest="emb_dim", type = int, help="size of sentence embedding")
    parser.add_argument("--num_epochs", dest="epochs", type = int, help="number of epochs to train")
    parser.add_argument("--result", dest="result", help="Name of result file")
    parser.add_argument("--errors", dest="errors", help="Name of result file")
    parser.add_argument("--batch", dest="batch", type = int, help="Batch size")
    parser.add_argument("--cuma", dest="cuma", help="Name for cumulative file")
    args, rest = parser.parse_known_args()

    make_dirs(args.model_name)
    make_dirs(args.result)

    print(f"**Is training**")

    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.model_name), exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_epochs = args.epochs
    best_accuracy = 0

    # Get batches
    train_batches, train_metadata = get_batch(args.train, device = device)
    test_batches, test_metadata = get_batch(args.test, batch_size=1, device = device)

    model = BasicClassifier(input_size = args.emb_dim, output_size=2)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    with open(args.result, "w") as file:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            input_len = 0
            
            # Train
            for input, label in zip(*train_batches):
                optimizer.zero_grad()
                input = input.to(device)
                # print(f"Input: {input.shape}")
                label = label.to(device)
                # print(f"Label: {label.shape}")

                # Output and loss
                output = model(input).squeeze(dim=1)
                # print(f"Output: {output.shape}")
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                input_len += label.size(0)
            
            print(f"Epoch: {epoch}, Loss: {running_loss / input_len}")
            file.write(f"Epoch: {epoch}, Loss: {running_loss / input_len}\n")
                
            # Eval
            print(f"Test batches: {len(test_batches)}")
            print(f"Test batches metadata: {len(test_metadata)}")
            accuracy, incorrect_texts = evaluate(model, test_batches, test_metadata, device)
            print(f"Accuracy: {accuracy:.2f}")
            file.write(f"Accuracy: {accuracy:.2f}\n")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), args.model_name)
        print("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))
        file.write("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))

    with open(args.errors, "w") as errors:
        errors.write(json.dumps(incorrect_texts[:10]))
        
    with open(args.cuma, "a") as cumulative:
        cumulative.write(f"{best_accuracy}\n")
        
    print("DONE")