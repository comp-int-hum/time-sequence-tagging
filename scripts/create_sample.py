import argparse
import jsonlines
import random
import os
from utility import make_dirs, open_file
import gzip
import sys
from tqdm import tqdm

def shuffle_file(sources, seed, sample_size):
    """
        Shuffle datapoints in source files and use reservoir sampling to reduce size.
    Args:
        sources (list): each element is a string filepath
        seed (int): seed for random shuffling
        sample_size (int): maximum sample size

    Returns:
        list: each element is a random line in the file
    """    
    random.seed(seed)
    data = []
    for file in sources:
        fp = open_file(file, "rt")
        # Algorithm R --> todo: substitute for Algorithm L
        for i, text in tqdm(enumerate(fp), desc = "Reading files in for shuffling"):
            if len(data) < sample_size:
                data.append(text)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    data[j] = text
        fp.close()
    random.shuffle(data)
    return data

# Downsample data to sample_size, collating from different filepaths (priority is given to items earlier in filepaths list)
def downsample_file(sources, output_path, sample_size):
    """Downsample data to sample_size, collating data from different sources with priority given to items 
       earlier in the sources list.

    Args:
        sources (list): where each element is a str filepath
        output_path (str): output_path as a string
        sample_size (int): maximum number of elements to retain during downsampling process
    """    
    with open(output_path, "wt") as output_file:
        total = 0
        for file in sources:
            with open_file(file, "rt") as fp:
                for line in tqdm(fp, desc = f"Processing lines in {file}"):
                    if total < sample_size:
                        output_file.write(line)
                        total += 1
                    else:
                        return

def get_data_size(filepath):
    with open_file(filepath, "rb") as input_file:
        return sum(1 for _ in input_file)

def write_data(filepath, data, start=None, end=None):
    with open(filepath, mode="w") as writer:
        for item in tqdm(data[start:end], desc="Writing out data in create sample"):
            writer.write(item)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs = "*", help="Filepaths containing data to be sampled")
    parser.add_argument("--output", dest="output", help="Output file name")
    parser.add_argument("--shuffle", dest="shuffle", type = int, help = "To shuffle or not")
    parser.add_argument("--sample_size", dest="sample_size", nargs = "?", default = sys.maxsize, type=int, help = "Max size")
    parser.add_argument("--seed", dest="seed", type = int, default = 0)
    args, rest = parser.parse_known_args()

    make_dirs(args.output)

    if args.shuffle:
        print(f"Seed: {args.seed}")
        data = shuffle_file(args.inputs, args.seed, args.sample_size)
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

# def read_shuffle_jsonl_file(filepaths, seed, sample_size):
#     random.seed(seed)
#     data = []
#     for file in filepaths:
#         fp = open_file(file, "rt")
#         # Algorithm R --> todo: substitute for Algorithm L
#         for i, text in enumerate(fp):
#             if len(data) < sample_size:
#                 data.append(text)
#             else:
#                 j = random.randint(0, i)
#                 if j < sample_size:
#                     data[j] = text
#         fp.close()
#     random.shuffle(data)
#     return data

# def shuffle_reduced_mem(filepaths, seed, sample_size):
#     random.seed(seed)
#     data_sizes = [get_data_size(file) for file in filepaths]
#     total_size = sum(data_sizes)
#     sampled_ids = random.sample(range(total_size), min(total_size, sample_size))
    
#     for file in filepaths:
#         fp = open_file(file, "rt")
#         for i, text in enumerate(fp):
#             if i in sampled_ids:
                
                
                
#         fp.close()
#     random.shuffle(data)
#     return data