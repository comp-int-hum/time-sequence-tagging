import argparse
import jsonlines
import random
import os
from utility import make_dirs, open_file
import gzip
import sys
from tqdm import tqdm

# Read up to sample_size from each file in filepaths
def read_shuffle_jsonl_file(filepaths, seed, sample_size):
    random.seed(seed)
    data = []
    for file in filepaths:
        fp = open_file(file, "rt")
        # Algorithm R --> todo: substitute for Algorithm L
        for i, text in enumerate(fp):
            if len(data) < sample_size:
                data.append(text)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    data[j] = text
        fp.close()
    random.shuffle(data)
    return data

# Read up to sample_size from each file in filepath
def downsample_file(filepaths, output_path, sample_size):
    with open(output_path, "wt") as output_file:
        total = 0
        for file in filepaths:
            with open_file(file, "rt") as fp:
                for line in tqdm(fp, desc = f"Processing lines in {file}"):
                    if total < sample_size:
                        output_file.write(line)
                        total += 1
                    else:
                        return

def get_data_size(filepath):
    reader = jsonlines.Reader(filename = open_file(filepath, "rb"))
    reader.close()
    return sum(1 for _ in reader)  

def write_data(filepath, data, start=None, end=None):
    with jsonlines.open(filepath, mode="wt") as writer:
        for item in data[start:end]:
            writer.write(item)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs = "*", help="File containing data to be sampled")
    parser.add_argument("--output", dest="output", help="Output file name")
    parser.add_argument("--shuffle", dest="shuffle", type = int, help = "To shuffle or not")
    parser.add_argument("--sample_size", dest="sample_size", default = sys.maxsize, type=int, help = "Max size")
    parser.add_argument("--seed", dest="seed", type = int, nargs = "*", default = 0)
    args, rest = parser.parse_known_args()

    make_dirs(args.output)

    if args.shuffle:
        data = read_shuffle_jsonl_file(args.inputs, args.seed, args.sample_size)
        write_data(args.output, data, end = min(args.sample_size, len(data)))

    else:
        print(f"Downsampling file to {args.sample_size}")
        downsample_file(args.inputs, args.output, args.sample_size)

    
    

# def float_type(arg):
#     try:
#         return float(arg)
#     except ValueError:
#         raise argparse.ArgumentTypeError(f"{arg} is not a valid float value")


# I'll rethink this file in the future.
            
# def read_shuffle_compressed_jsonl_file(filepaths, seed, sample_size, file_size):
#     print("Start shuffle")
#     random.seed(seed)
#     data = []
#     for file in filepaths:
#         size = get_data_size(file) if not file_size else file_size
#         sample = random.sample(range(size), min(size, sample_size))
#         with gzip.open(file, mode="rt") as gzipfile:
#             reader = jsonlines.Reader(gzipfile)
#             for i, text in enumerate(reader):
#                 if i in sample:
#                     data.append(text)
#     random.shuffle(data)
#     return data