import jsonlines
from collections import OrderedDict
import argparse
import random
from utility import make_dirs
import gzip
import re
from utility import parse_labels

def random_subsequence(sequence, minimum_len, maximum_len):
    # Validate min max
    max_len = min(maximum_len, len(sequence)) 
    min_len = max(minimum_len, 1)

    if min_len > max_len:
        print(f"Error creating sequence: min: {minimum_len} - max: {maximum_len}")
        return None

    # Generate random seq_len and start
    sub_seq_len = random.randint(min_len, max_len)
    start_idx = random.randint(0, len(sequence) - sub_seq_len)

    return sequence[start_idx : start_idx + sub_seq_len]

# Based on matched_idxs, sample
# Sequence: ((tag, embedding), original_text)
def random_biased_subsequence(sequence,  matched_idxs, min_len, max_len):
    max_len = min(max_len, len(sequence)) 
    min_len = max(min_len, 1)

    if min_len > max_len or not matched_idxs:
        return None
    
    sub_seq_len = random.randint(min_len, max_len)
    chosen_match = random.choice(matched_idxs)
    start_idx = random.randint(max(0, chosen_match - sub_seq_len), chosen_match)
    return sequence[start_idx: start_idx + sub_seq_len]
    

def get_matching_idxs(sequence, label_type):
    _, labels, _ = unpack_sequence_data(sequence)
    return [i for i, label in enumerate(labels) if label == label_type]

def unpack_sequence_data(sequence):
    data_seq, text_seq = zip(*sequence)
    labels, embeds = zip(*data_seq)
    return text_seq, labels, embeds

def sample_from_beginning(sequence, minimum_len, maximum_len):
    # Validate min max
    max_len = min(maximum_len, len(sequence)) 
    min_len = max(minimum_len, 1)

    if min_len > max_len:
        print(f"Error creating sequence: min: {minimum_len} - max: {maximum_len}")
        return None

    # Generate random seq_len
    sub_seq_len = random.randint(min_len, max_len)

    return sequence[:sub_seq_len]

# Todo: fix this
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
    parser.add_argument("--output", dest="output", help="Name for datapoints file")
    parser.add_argument("--min_seq", type=int, default = 1, dest="min_len", help="Min len for sequence")
    parser.add_argument("--max_seq", type=int, default = 10000, dest="max_len", help="Max len for sequence")
    parser.add_argument("--sample_method", choices = ["from_beginning", "from_chapter_beginning", "random_subseq"], help = "Type of sampling: from beginning, from chapter_beginning, random_subseq")
    parser.add_argument("--samples", type=int, dest="samples", help="Number of samples to take")
    parser.add_argument("--seed", dest="seed", type=int)
    args, rest = parser.parse_known_args()

    random.seed(args.seed)

    make_dirs(args.output)
    
    print("**Creating datapoints**")
    
    with gzip.open(args.input, "r") as input_file, gzip.open(args.output, mode="wt") as output_file:
        with jsonlines.Reader(input_file) as input, jsonlines.Writer(output_file) as writer:
            # For line in jsonlines
            data = [] # datapoints for current book
            print(f"max: {args.max_len}")
            print(f"min: {args.min_len}")
            for idx, doc in enumerate(input):
                print(f"IDX: {idx}")
                
                zipped_lst = list(zip(
                    doc["paragraph_labels"], 
                    doc["chapter_labels"], 
                    doc["text"], 
                    doc["embeddings"]
                ))

                for _ in range(args.samples):
                    match args.sample_method:
                        case "from_beginning":
                            sample_seq = sample_from_beginning(zipped_lst, args.min_len, args.max_len)
                        case "from_chapter_beginning":
                            sample_seq = sample_from_chapter_beginnings(zipped_lst, args.min_len, args.max_len)
                        case "random_subseq":
                            sample_seq = random_subsequence(zipped_lst, args.min_len, args.max_len)
                        case _:
                            raise ValueError("Did not match sampling method")
                    if not sample_seq:
                        break
                    paragraph_labels, chapter_labels, text, embeddings = zip(*sample_seq)
                    datapoint = {"id" : doc["id"],
                                "granularity": doc["granularity"],
                                "paragraph_labels": paragraph_labels,
                                "chapter_labels": chapter_labels,
                                "text": text,
                                "embeddings": embeddings}
                    data.append(datapoint)
                

            random.shuffle(data)
            print(f"Total data length : {len(data)}")

            # Write resulting data
            for d in data:
                writer.write(d)