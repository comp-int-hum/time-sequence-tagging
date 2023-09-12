import argparse
import jsonlines
import random
import os
from utility import make_dirs
# def float_type(arg):
#     try:
#         return float(arg)
#     except ValueError:
#         raise argparse.ArgumentTypeError(f"{arg} is not a valid float value")


# I'll rethink this file in the future.

def read_shuffle_jsonl_file(filepaths, seed, sample_size):
    random.seed(seed)
    data = []
    for file in filepaths:
        size = get_data_size(file)
        sample = random.sample(range(size), min(size, sample_size))
        with jsonlines.open(file, mode="r") as reader:
            for i, text in enumerate(reader):
                if i in sample:
                    data.append(text)
    random.shuffle(data)
    return data

def get_data_size(filepath):
    with jsonlines.open(filepath, mode = "r") as reader:
        return sum(1 for line in reader)

def write_data(filepath, data, start=0, end=None):
    with jsonlines.open(filepath, mode="w") as writer:
        for item in data[start:end]:
            writer.write(item)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs = "*", help="File containing data to be sampled")
    parser.add_argument("--output", dest="output", help="Output file name")
    parser.add_argument("--data_size", type=int, dest="max_data_size")
    parser.add_argument("--seed", dest="seed", type = int)
    args, rest = parser.parse_known_args()

    make_dirs(args.output)

    print(f"Shuffling data")
    data = read_shuffle_jsonl_file(args.inputs, args.seed, args.data_size)
    write_data(args.output, data, end = min(args.data_size, len(data)))
        
            
