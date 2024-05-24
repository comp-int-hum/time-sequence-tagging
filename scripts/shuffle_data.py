import argparse
import jsonlines
import random
from utility import make_dirs
import gzip
import json

            
def get_data_size(filepath):
    with gzip.open(filepath, mode = "r") as input_file:
        reader = jsonlines.Reader(input_file)
        return sum(1 for _ in reader)     

def generate_splits(datapath, output_paths, all_samples):
    num_experiments = len(all_samples)
    with gzip.open(datapath, "r") as data_file:
        reader = jsonlines.Reader(data_file)
        for i, text in enumerate(reader):
            for num, (train, test) in enumerate(all_samples):
                if i in train:
                    output_paths[num].write(json.dumps(text) + "\n")
                elif i in test:
                    output_paths[num_experiments + num].write(json.dumps(text) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="datapath", nargs = 1, help="Primary training file")
    parser.add_argument("--outputs", dest="outputs", nargs="*", help="Output filepaths")
    parser.add_argument("--sample_size", type=int, dest="sample_size")
    parser.add_argument("--split_ratio", type=float, dest="ratio")
    parser.add_argument("--cd", dest="cd", nargs="?", default = None, help = "If it exists, the cross_domain test file")
    parser.add_argument("--seed", dest="seed", type=int, help = "Seed for random shuffle")
    args, rest = parser.parse_known_args()

    if len(args.outputs) % 2:
        raise argparse.ArgumentTypeError(f"Incorrect number of filepaths")
    
    random.seed(args.seed)
    
    print(f"***************** BEGIN SHUFFLE **********************")

    # Make output paths
    output_paths = []
    for output in args.outputs:
        make_dirs(output)
        output_paths.append(gzip.open(output, "wt"))
            
    # Calculate sample sizes for source document
    num_repeat = len(args.outputs) // 2
    source_size = get_data_size(args.datapath[0])
    source_sample_size = source_size if not args.sample_size else args.sample_size # default source_sample_size is entire sample
    source_sample_size = min(source_sample_size, source_size)
    tr_size = int(args.ratio * source_sample_size)

    # Build source document samples
    source_samples = []

    for i in range(num_repeat):
        curr_sample = random.sample(range(source_size), source_sample_size)
        train = set(curr_sample[:tr_size])
        test = set(curr_sample[tr_size:]) if not args.cd else []
        source_samples.append((train, test))

    # Write to output source document splits
    generate_splits(args.datapath[0], output_paths, source_samples)

    # Build cross domain document samples if cd flag set
    if args.cd:

        # Calculate sample sizes for cross domain document
        cross_size = get_data_size(args.cd)
        cross_sample_size = min((1-args.ratio) * source_sample_size, cross_size) # set to be at most the same as test size for non-cross-domain

        # Build cross domain document samples
        cross_samples = []
        for i in range(num_repeat):
            curr_sample = random.sample(range(cross_size), cross_sample_size)
            train = []
            test = set(curr_sample)
            cross_samples.append((train, test))

        # Write to output cross domain document splits
        generate_splits(args.cd, output_paths, cross_samples)

    for output in output_paths:
        output.close()
    

    print(f"Shuffling data")
    
    
    
        
            
