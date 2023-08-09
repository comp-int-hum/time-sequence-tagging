import argparse
from transformers import BertModel, BertConfig
import jsonlines
import random

# def float_type(arg):
#     try:
#         return float(arg)
#     except ValueError:
#         raise argparse.ArgumentTypeError(f"{arg} is not a valid float value")


# I'll rethink this file in the future.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--output", dest="output", nargs=2, help="Output files")
    parser.add_argument("--data_size", type=int, dest="data_size")
    parser.add_argument("--split_ratio", type=float, dest="ratio")
    args, rest = parser.parse_known_args()

    data = []

    print(f"Shuffling data")
    with jsonlines.open(args.input, mode="r") as reader, jsonlines.open(args.output[0], mode="w") as train_writer, jsonlines.open(args.output[1], mode="w") as test_writer:
        # For line in jsonlines
        for text in reader:
            data.append(text)
        random.shuffle(data)

        data_size = min(args.data_size, len(data))
        tr_size = int(0.8 * data_size)
        ev_size = int(0.2 * data_size)
        print(f"Training size: {tr_size}")
        print(f"Eval size: {ev_size}")

        for i in range(data_size):
            if i < tr_size:
                train_writer.write(data[i])
            else:
                test_writer.write(data[i])
        
            
