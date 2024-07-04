import argparse
import jsonlines
import random
from utility import make_dirs, open_file
import gzip
import json
      
def get_data_size(filepath):
    reader = jsonlines.Reader(open_file(filepath, "rt"))
    return sum(1 for _ in reader)     

def generate_splits(datapath, output_paths, folds):
    num_folds = len(folds)
    with gzip.open(datapath, "r") as data_file:
        reader = jsonlines.Reader(data_file)
        for i, text in enumerate(reader):
            for num, (train, dev, test) in enumerate(folds):
                if i in train:
                    output_paths[num].write(json.dumps(text) + "\n")
                elif i in dev:
                    output_paths[num_folds + num].write(json.dumps(text) + "\n")
                elif i in test:
                    output_paths[2 * num_folds + num].write(json.dumps(text) + "\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="datapath", nargs = 1, help="Primary training file")
    parser.add_argument("--outputs", dest="outputs", nargs="*", help="Output filepaths")
    parser.add_argument("--sample_size", type=int, nargs = "?", dest="sample_size")
    parser.add_argument("--split_ratio", nargs = 3, type=float, dest="ratio")
    parser.add_argument("--cd", dest="cd", nargs="?", default = None, help = "If it exists, the cross_domain test file")
    parser.add_argument("--seed", dest="seed", type=int, help = "Seed for random shuffle")
    args, rest = parser.parse_known_args()

    if len(args.outputs) % 3:
        raise argparse.ArgumentTypeError(f"Incorrect number of filepaths")
    
    random.seed(args.seed)
    
    print(f"***************** BEGIN SHUFFLE **********************")

    # Make output paths
    output_paths = []
    for output in args.outputs:
        make_dirs(output)
        output_paths.append(gzip.open(output, "wt"))
            
    # Calculate number of folds and max sample size for source document
    num_folds = len(args.outputs) // 3
    source_size = get_data_size(args.datapath[0])
    source_sample_size = source_size if not args.sample_size else args.sample_size # default source_sample_size is entire sample
    source_sample_size = min(source_sample_size, source_size)

    # Get train and dev sizes
    tr_size = int(args.ratio[0] * source_sample_size)
    dev_size = int(sum(args.ratio[0:2]) * source_sample_size)
    
    # Build source document samples
    folds = []

    for i in range(num_folds):
        curr_sample = random.sample(range(source_size), source_sample_size)
        train = set(curr_sample[:tr_size])
        dev = set(curr_sample[tr_size : dev_size]) if not args.cd else []
        test = set(curr_sample[dev_size:]) if not args.cd else []
        folds.append((train, dev, test))

    # Write to output source document splits
    generate_splits(args.datapath[0], output_paths, folds)

    # Build cross domain document samples if cd flag set
    if args.cd:

        # Calculate sample sizes for cross domain document
        cross_size = get_data_size(args.cd)
        cross_sample_size = min((1-args.ratio[0]) * source_sample_size, cross_size) # set to be at most the same as test size for non-cross-domain
        dev_size = int((args.ratio[1] / sum(args.ratio[1:3]) * cross_sample_size))
        # Build cross domain document samples
        cross_samples = []
        for i in range(num_folds):
            curr_sample = random.sample(range(cross_size), cross_sample_size)
            train = []
            dev = set(curr_sample[:dev_size])
            test = set(curr_sample[dev_size:])
            cross_samples.append((train, dev, test))

        # Write to output cross domain document splits
        generate_splits(args.cd, output_paths, cross_samples)

    for output in output_paths:
        output.close()
    

    print(f"Shuffling data")
    
    
    
        
            
