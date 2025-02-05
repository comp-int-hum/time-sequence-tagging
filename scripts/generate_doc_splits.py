import jsonlines
from collections import OrderedDict
import argparse
import random
import gzip
import re
import logging
from utility import parse_labels
from tqdm import tqdm
import sys

logger = logging.getLogger("generate_splits.py")

def random_subsequence(sequence, minimum_len, maximum_len):
    """Sample a random subsequence between minimum_len and maximum_len from the overall sequence.

    Args:
        sequence (list): zipped list of all sequences that is to be sampled from
        minimum_len (int): minimum length of subsequence to sample
        maximum_len (int): maximum length of subsequence to sample

    Returns:
        list: sampled subsequence
    """    
    # Validate min max
    max_len = min(maximum_len, len(sequence)) 
    min_len = max(minimum_len, 1)

    if min_len > max_len:
        print(f"Error creating sequence: min: {minimum_len} - max: {maximum_len}")
        return []

    # Generate random seq_len and start
    sub_seq_len = random.randint(min_len, max_len)
    start_idx = random.randint(0, len(sequence) - sub_seq_len)

    return sequence[start_idx : start_idx + sub_seq_len]

def random_subsequences(sequence, minimum_len, maximum_len, num_samples):
    """Sample a random subsequence between minimum_len and maximum_len from the overall sequence.

    Args:
        sequence (list): zipped list of all sequences that is to be sampled from
        minimum_len (int): minimum length of subsequence to sample
        maximum_len (int): maximum length of subsequence to sample

    Returns:
        list: sampled subsequence
    """    
    # Validate min max
    max_len = min(maximum_len, len(sequence)) 
    min_len = max(minimum_len, 1)

    if min_len > max_len:
        print(f"Error creating sequence: min: {minimum_len} - max: {maximum_len}")
        return []

    # Generate random sample_length and start_idx
    sample_length = random.randint(min_len, max_len)
    start_idx = random.randint(0, min(sample_length, len(sequence) - sample_length))
    
    # Generate samples
    samples = []
    for i in range(start_idx, len(sequence), sample_length):
        sample = sequence[i:i+sample_length]
        if len(sample) == sample_length:
            samples.append(sample)

    return random.sample(samples, min(len(samples), num_samples))


def sample_from_beginning(sequence, minimum_len, maximum_len):
    """Sample a random subsequence starting from the beginning between minimum_len and maximum_len.

    Args:
        sequence (list): zipped list of all sequences that is to be sampled from
        minimum_len (int): minimum length of subsequence to sample
        maximum_len (int): maximum length of subsequence to sample

    Returns:
        list: sampled subsequence
    """    
    # Validate min max
    max_len = min(maximum_len, len(sequence)) 
    min_len = max(minimum_len, 1)

    if min_len > max_len:
        print(f"Error creating sequence: min: {minimum_len} - max: {maximum_len}")
        return []

    # Generate random seq_len
    sub_seq_len = random.randint(min_len, max_len)

    return [sequence[:sub_seq_len]]

# TODO: fix this
def sample_from_chapter_beginnings(sequence, minimum_len, maximum_len):
    max_len = min(maximum_len, len(sequence)) 
    min_len = max(minimum_len, 1)

    if min_len > max_len:
        print(f"Error creating sequence: min: {minimum_len} - max: {maximum_len}")
        return None
    
    # Fix this
    label_lst, _ = zip(*sequence)
    str_rep = "".join([str(label) for label in label_lst])
    start_idxs = [matched.end() for matched in re.finditer(r"11", str_rep)] + [0]
    
    start_idx = random.choice(start_idxs)
    sub_seq_len = random.randint(min_len, max_len)
    return sequence[start_idx: start_idx + sub_seq_len]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Sequence file")
    parser.add_argument("--train", dest="train", help="Name for datapoints file")
    parser.add_argument("--dev", dest="dev", help="Name for datapoints file")
    parser.add_argument("--test", dest="test", help="Name for datapoints file")
    parser.add_argument("--min_len", type=int, default = 1, dest="min_len", help="Min len for sequence")
    parser.add_argument("--max_len", type=int, default = 10000, dest="max_len", help="Max len for sequence")
    parser.add_argument(
        "--sample_method",
        choices = ["from_beginning", "from_chapter_beginning", "random_subseq"],
        help = "Type of sampling: from beginning, from chapter_beginning, random_subseq"
    )
    parser.add_argument("--samples_per_document", type=int, dest="samples_per_document", help="Number of samples to take from each document")
    parser.add_argument("--train_proportion", type=float, default=0.8)
    parser.add_argument("--dev_proportion", type=float, default=0.1)
    parser.add_argument("--test_proportion", type=float, default=0.1)
    parser.add_argument("--random_seed", dest="random_seed", type=int)
    args, rest = parser.parse_known_args()

    if args.random_seed != None:
        random.seed(args.random_seed)

    logging.basicConfig(level=logging.INFO)
        
    logger.info("Creating datapoints")
    
    with gzip.open(args.input, "r") as ifd, gzip.open(args.train, mode="wt") as train_ofd, gzip.open(args.dev, mode="wt") as dev_ofd, gzip.open(args.test, mode="wt") as test_ofd:
        with jsonlines.Reader(ifd) as input_reader, jsonlines.Writer(train_ofd) as train_writer, jsonlines.Writer(dev_ofd) as dev_writer, jsonlines.Writer(test_ofd) as test_writer:
            counter = 0
            for idx, doc in tqdm(enumerate(input_reader)):
                # zipped_lst = list(
                #     zip(
                #         doc["paragraph_labels"], 
                #         doc["chapter_labels"], 
                #         doc["flattened_sentences"], 
                #         doc["flattened_embeddings"],
                #         doc["hierarchical_labels"]
                #     )
                # )
                rv = random.random()
                if rv < args.train_proportion:
                    split_writer = train_writer
                elif rv < args.train_proportion + args.dev_proportion:
                    split_writer = dev_writer
                else:
                    split_writer = test_writer
                # paragraph_labels, chapter_labels, flattened_sentences, flattened_embeddings, hierarchical_labels = zip(*zipped_lst)
                # datapoint = {
                #     "metadata": doc["metadata"],
                #     "granularity": doc["granularity"],
                #     "paragraph_labels": paragraph_labels,
                #     "chapter_labels": chapter_labels,
                #     "flattened_sentences": flattened_sentences,
                #     "flattened_embeddings": flattened_embeddings,
                #     "hierarchical_labels": hierarchical_labels
                # }
                # split_writer.write(datapoint)
                split_writer.write(doc)
                counter += 1
            print(f"Total data length : {counter}")
                
                
# # Based on matched_idxs, sample
# # Sequence: ((tag, embedding), original_text)
# def random_biased_subsequence(sequence,  matched_idxs, min_len, max_len):
#     max_len = min(max_len, len(sequence)) 
#     min_len = max(min_len, 1)

#     if min_len > max_len or not matched_idxs:
#         return []
    
#     sub_seq_len = random.randint(min_len, max_len)
#     chosen_match = random.choice(matched_idxs)
#     start_idx = random.randint(max(0, chosen_match - sub_seq_len), chosen_match)
#     return sequence[start_idx: start_idx + sub_seq_len]
    

# def get_matching_idxs(sequence, label_type):
#     _, labels, _ = unpack_sequence_data(sequence)
#     return [i for i, label in enumerate(labels) if label == label_type]

# def unpack_sequence_data(sequence):
#     data_seq, text_seq = zip(*sequence)
#     labels, embeds = zip(*data_seq)
#     return text_seq, labels, embeds

# for _ in range(args.samples):
    # match args.sample_method:
    #     case "from_beginning":
    #         sample_seq = sample_from_beginning(zipped_lst, args.min_len, args.max_len)
    #     case "from_chapter_beginning":
    #         sample_seq = sample_from_chapter_beginnings(zipped_lst, args.min_len, args.max_len)
    #     case "random_subseq":
    #         sample_seq = random_subsequence(zipped_lst, args.min_len, args.max_len)
    #     case _:
    #         raise ValueError("Did not match sampling method")
