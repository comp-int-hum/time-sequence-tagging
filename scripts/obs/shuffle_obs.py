import argparse
import jsonlines
import random
import os
from create_sample import read_shuffle_compressed_jsonl_file, write_compressed_data
from utility import make_dirs

def read_shuffle_compressed_jsonl_file(filepaths, sample_size, max_file_size, seed):
    print("Start shuffle")
    random.seed(seed)
    data = []
    sample_ids = random.sample(range(len(filepaths) * max_file_size), sample_size)
    offset = 0
    for file in filepaths:
        with gzip.open(file, mode="rt") as gzipfile:
            reader = jsonlines.Reader(gzipfile)
            for i, text in enumerate(reader):
                if (i + offset) in sample_ids:
                    data.append(text)
            offset+=i
    random.shuffle(data)
    return data

# def write_compressed_data(filepath, data, start=0, end=None):
#     with gzip.open(filepath, mode="w") as gzipfile:
#         writer = jsonlines.Writer(gzipfile)
#         for item in data[start:end]:
#             writer.write(item)

# def train_test_split(source_data, test_data, ratio, train_path, test_path):
#     source_size = len(source_data)
#     tr_size = int(ratio * source_size)
#     if not len(test_data):
#         test_data = source_data[tr_size : ]
#     with gzip.open(train_path, "w") as train_file, gzip.open(test_path, "w") as test_file:
#         for item in source_data[0 : tr_size]:
#             train_file.write(json.dumps(item))
#         for item in test_data:
#             test_file.write(json.dumps(item))


# def read_shuffle_compressed_jsonl_file(filepaths, sample_size, max_file_size, seed):
#     print("Start shuffle")
#     random.seed(seed)
#     data = []
#     sample_ids = random.sample(range(len(filepaths) * max_file_size), sample_size)
#     offset = 0
#     for file in filepaths:
#         with gzip.open(file, mode="rt") as gzipfile:
#             reader = jsonlines.Reader(gzipfile)
#             for i, text in enumerate(reader):
#                 if (i + offset) in sample_ids:
#                     data.append(text)
#             offset+=i
#     random.shuffle(data)
#     return data

# def generate_sample_indices(num_repeat, data_size, sample_size, ratio, cd):
#     tr_size = int(args.ratio * source_sample_size)

# def generate_splits(datapath, output_paths, sample_size=0, ratio=0.8):
#     num_repeat = len(output_paths)
#     data_size = get_data_size(datapath)
#     sample_size = data_size if not sample_size else sample_size
#     sample_size = min(sample_size, data_size)
#     tr_size = int(ratio * sample_size)

#     all_samples = []

#     for i in range(num_repeat):
#         curr_sample = random.sample(range(data_size), sample_size)
#         all_samples.append(set(curr_sample))
    
#     outputs = []
#     for output in output_paths:
#         outputs.append(gzip.open(output, "w"))
    
#     with gzip.open(datapath, "r") as data_file:
#         reader = jsonlines.Reader(data_file)
#         for i, text in enumerate(reader):
#             for num, sample in enumerate(all_samples):
#                 if i in sample:
#                     outputs[num].write(json.dumps(text) + "\n")

#     for output in outputs:
#         output.close()       


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs = "*", help="File containing gutenberg & women writers project")
    parser.add_argument("--output", dest="output", nargs=2, help="Output files")
    parser.add_argument("--sample_size", type=int, dest="sample_size")
    parser.add_argument("--split_ratio", type=float, dest="ratio")
    parser.add_argument("--cd", dest="cd", nargs="?", help = "If exists, is cross_domain test file")
    parser.add_argument("--seed", dest="seed", type=int, help = "Seed for random shuffle")
    parser.add_argument("--max_file_size", dest="file_size", type=int, default=0, help="Largest possible encode file size")
    args, rest = parser.parse_known_args()

    for output in args.output:
        make_dirs(output)

    print(f"Shuffling data")
    train_data = read_shuffle_compressed_jsonl_file(args.inputs, int(args.sample_size * args.ratio), args.file_size, args.seed)
    
    if args.cd and os.path.isfile(args.cd):
        print(f"File path found {args.cd}")
        test_data = read_shuffle_compressed_jsonl_file(args.cd, int(args.sample_size * (1 - args.ratio)), args.seed)
        tr_size = int(min(args.ratio * args.sample_size, len(train_data)))
        ev_size = int(min((1-args.ratio) * args.sample_size, len(test_data)))
        write_compressed_data(args.output[0], train_data, end=tr_size)
        write_compressed_data(args.output[1], test_data, end=ev_size)
    else:
        print(f"No file path found, splitting data into train + test")
        data_size = min(args.sample_size, len(train_data))
        tr_size = int(args.ratio * data_size)
        write_compressed_data(args.output[0], train_data, end=tr_size)
        write_compressed_data(args.output[1], train_data, tr_size + 1, data_size)