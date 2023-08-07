import argparse
from transformers import BertModel, BertConfig
import jsonlines
import random


# I'll rethink this file in the future.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--output", dest="output", help="Output files")
    parser.add_argument("--train_files", type=int, dest="num_files")
    args, rest = parser.parse_known_args()

    data = []
    with jsonlines.open(args.input, mode="r") as reader, jsonlines.open(args.output, mode="w") as writer:
        # For line in jsonlines
        for text in reader:
            data.append(text)
        random.shuffle(data)

        minimum = min(args.num_files, len(data))
        for i in range(minimum):
            writer.write(data[i])
        
            