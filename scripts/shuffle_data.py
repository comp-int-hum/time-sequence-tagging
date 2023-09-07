import argparse
import jsonlines
import random
import os
# def float_type(arg):
#     try:
#         return float(arg)
#     except ValueError:
#         raise argparse.ArgumentTypeError(f"{arg} is not a valid float value")


# I'll rethink this file in the future.

def read_shuffle_jsonl_file(filepaths):
    data = []
    for file in filepaths:
        with jsonlines.open(file, mode="r") as reader:
            for text in reader:
                data.append(text)
    random.shuffle(data)
    return data

def write_data(filepath, data, start=0, end=None):
    with jsonlines.open(filepath, mode="w") as writer:
        for item in data[start:end]:
            writer.write(item)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs = "*", help="File containing gutenberg & women writers project")
    parser.add_argument("--output", dest="output", nargs=2, help="Output files")
    parser.add_argument("--max_data_size", type=int, dest="max_data_size")
    parser.add_argument("--split_ratio", type=float, dest="ratio")
    parser.add_argument("--cd", dest="cd", nargs="?", help = "If exists, is cross_domain test file")
    args, rest = parser.parse_known_args()

    

    print(f"Shuffling data")
    train_data = read_shuffle_jsonl_file(args.input)
    
    if os.path.isfile(args.cd):
        test_data = read_shuffle_jsonl_file(args.cd)
        tr_size = int(min(args.ratio * args.max_data_size, len(train_data)))
        ev_size = int(min((1-args.ratio) * args.max_data_size, len(test_data)))
        write_data(args.output[0], train_data, end=tr_size)
        write_data(args.output[1], test_data, end=ev_size)
    else:
        data_size = min(args.max_data_size, len(train_data))
        tr_size = int(args.ratio * data_size)
        write_data(args.output[0], train_data, end=tr_size)
        write_data(args.output[1], train_data, tr_size + 1, data_size)
        
            
