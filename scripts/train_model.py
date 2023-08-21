import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import json

class BasicBinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        print(f"Input size: {input_size}")
        self.fc1 = nn.Linear(input_size, 256)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in")
        self.fc2 = nn.Linear(256, 128)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in")
        self.fc3 = nn.Linear(128,1)
        nn.init.kaiming_normal_(self.fc3.weight, mode="fan_in")
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, data):
        # print(f"First: {first.shape}")
        # print(f"Second: {second.shape}")
        # x = torch.cat((first, second), dim = 0).to("cuda")
        # print(f"Cat {x.shape}")
        x = self.fc1(data)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def get_batch(datapoints, batch_size = 16):
    data_batch = []
    label_batch = []

    curr_data_batch = []
    curr_label_batch = []
    for i, datapoint in enumerate(datapoints):
        if datapoint["embeddings"]:
            raise ValueError("missing embedding")
        curr_data_batch.append(datapoint["embeddings"])
        curr_label_batch.append(float(datapoint["positive"]))
        
        if len(curr_data_batch) == batch_size:
            data_batch.append(torch.tensor(curr_data_batch))
            label_batch.append(torch.tensor(curr_label_batch))
            curr_data_batch = []
            curr_label_batch = []
    if curr_data_batch:
        data_batch.append(torch.tensor(curr_data_batch))
        label_batch.append(torch.tensor(curr_label_batch))

    return data_batch, label_batch

def evaluate(model, batches):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input, labels in batches:
            input.to(device)
            labels.to(device)

            # Output and loss
            output = model(input)
            correct += (torch.abs(labels - output) < 0.5).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", help="name of training datapoints file")
    parser.add_argument("--eval", dest="eval", help="name of test datapoints file")
    parser.add_argument("--model_name", dest="model_name", help="Name of best model")
    parser.add_argument("--emb_dim", dest="emb_dim", type = int, help="size of sentence embedding")
    parser.add_argument("--num_epochs", dest="epochs", type = int, help="number of epochs to train")
    parser.add_argument("--result", dest="result", help="Name of result file")
    parser.add_argument("--batch", dest="batch", type = int, help="Batch size")
    args, rest = parser.parse_known_args()

    print(f"Is training")

    torch.cuda.empty_cache()

    device = "cuda"

    model = BasicBinaryClassifier(input_size = args.emb_dim)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = args.epochs
    best_accuracy = 0
    with open(args.result, "w") as file:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            input_len = 0

            # Get batches
            with jsonlines.open(args.train, 'r') as input:
                batches = get_batch(input)
            
            # Train
            for input, label in batches:
                optimizer.zero_grad()

                input.to(device)
                label.to(device)

                # Output and loss
                output = model(input)

                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                input_len += label.size(0)
            
            print(f"Epoch: {epoch}, Loss: {running_loss / input_len}")
            file.write(f"Epoch: {epoch}, Loss: {running_loss / input_len}")

            # Eval
            accuracy = evaluate(model, batches)
            print(f"Accuracy: {accuracy:.2f}%")
            file.write(f"Accuracy: {accuracy:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), args.model_name)

        print("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))
        file.write("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))
