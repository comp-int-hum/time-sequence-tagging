import torch.nn.functional as F
import torch.nn as nn
class SequenceTagger(nn.Module):

    def __init__(self, input_size, hidden_dim, num_classes):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        super(SequenceTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim, num_layers = 1, batch_first = True, bidirectional = True)
        self.output = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, sentence_embeds):
        lstm_out, _ = self.lstm(sentence_embeds) # lstm_out is (batch_size, seq_len, 2 * hidden_dim)
        outputs = self.output(lstm_out)
        return outputs
    


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
    train_batches, train_metadata, train_size = get_batch(args.train, device = device)
    test_batches, test_metadata, test_size = get_batch(args.test, batch_size=1, device = device)

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
                label = label.to(device)
                output = model(input).squeeze(dim=1)
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
        file.write(f"Train size: {train_size} \n")
        file.write(f"Test size: {test_size} \n")
        file.write(f"Number of incorrect texts: {len(incorrect_texts)}")
        file.write("\nBest Performing Model achieves dev pearsonr of : %.3f" % (best_accuracy))

    with open(args.errors, "w") as errors:
        errors.write(json.dumps(incorrect_texts))
        
    with open(args.cuma, "a") as cumulative:
        cumulative.write(f"{best_accuracy}\n")
        
    print("DONE")