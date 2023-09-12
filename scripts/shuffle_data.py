import argparse
import jsonlines
import random
import os
from create_sample import read_shuffle_jsonl_file, write_data
from utility import make_dirs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs = "*", help="File containing gutenberg & women writers project")
    parser.add_argument("--output", dest="output", nargs=2, help="Output files")
    parser.add_argument("--sample_size", type=int, dest="sample_size")
    parser.add_argument("--split_ratio", type=float, dest="ratio")
    parser.add_argument("--cd", dest="cd", nargs="?", help = "If exists, is cross_domain test file")
    parser.add_argument("--seed", dest="seed", type=int, help = "Seed for random shuffle")
    args, rest = parser.parse_known_args()

    for output in args.output:
        make_dirs(args.output)

    print(f"Shuffling data")
    train_data = read_shuffle_jsonl_file(args.inputs, args.seed)
    
    if args.cd and os.path.isfile(args.cd):
        print(f"File path found {args.cd}")
        test_data = read_shuffle_jsonl_file(args.cd, args.seed)
        tr_size = int(min(args.ratio * args.sample_size, len(train_data)))
        ev_size = int(min((1-args.ratio) * args.sample_size, len(test_data)))
        write_data(args.output[0], train_data, end=tr_size)
        write_data(args.output[1], test_data, end=ev_size)
    else:
        print(f"No file path found, using train data")
        data_size = min(args.sample_size, len(train_data))
        tr_size = int(args.ratio * data_size)
        write_data(args.output[0], train_data, end=tr_size)
        write_data(args.output[1], train_data, tr_size + 1, data_size)
        
            
