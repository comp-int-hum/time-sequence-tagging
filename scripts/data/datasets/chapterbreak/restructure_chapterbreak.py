import argparse
import torch
import gzip
from tqdm import tqdm
import json
import nltk
import gzip
from utils.utility import make_parent_dirs_for_files
import re
import random

def filter_strings(strings, patterns):
    to_ret = []
    for s in strings:
        if not any(pattern.match(s) for pattern in patterns):
            to_ret.append(s)
    
    return to_ret

def get_hierarchical_labels(context_len, pos_len):
    total_len = context_len + pos_len
    hierarchical_labels = [0] * total_len

    if context_len > 0:
        hierarchical_labels[context_len - 1] = 1
    if context_len < total_len:
        hierarchical_labels[context_len] = 1

    return {"chapters": hierarchical_labels}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Chapterbreak file")
    parser.add_argument("--data_output", dest="data_output", help="Output file")
    parser.add_argument("--label_output", dest="label_output", help = "Label output file")
    parser.add_argument("--splits", dest = "splits", nargs = "+", default = ["pg19", "ao3"], help = "Which split(s) of the data to include")
    parser.add_argument("--title_filters", dest="title_filters", nargs = "*", default = [r"^[^a-zA-Z0-9]*[A-Z]+(?:'?[A-Z]+)?(\s[A-Z]+(?:'?[A-Z]+)?)*[^a-zA-Z0-9]*$"], help = "Sentence filter patterns")
    args, rest = parser.parse_known_args()
    
    make_parent_dirs_for_files([args.data_output, args.label_output])
    
    with open(args.input, "rt") as input_file:
        data = json.load(input_file)
    
    compiled_filters = [re.compile(filter) for filter in args.filters]
    
    outputs = []
    key = 0
    for split_name in args.splits:
        data_split = data[split_name]
        for text_id, text in tqdm(list(data_split.items()), desc = f"Iterating over texts in split: {split_name}"):
            for idx, triplet in enumerate(text):
                context = triplet["ctx"]
                pos = triplet["pos"]
                context_sentences = filter_strings(nltk.sent_tokenize(context), compiled_filters)
                pos_sentences = filter_strings(nltk.sent_tokenize(pos), compiled_filters)
                data = {
                        "metadata": {
                            "title": text_id,
                            "author": "",
                            "text_id": text_id,
                            "segment_num": idx,
                            "year": None,
                            "source": f"chapterbreak_{split_name}"
                        },
                        "content": context_sentences + pos_sentences,
                        "key": key
                }
                annotations = {
                        "base_unit": "sentences",
                        "labels": get_hierarchical_labels(len(context_sentences), len(pos_sentences)),
                        "hierarchy_order": ["sentences", "chapters"],
                        "source": f"chapterbreak_{split_name}",
                        "key": key
                }

                key += 1
                
                outputs.append({"data": data, "annotations": annotations})

    random.shuffle(outputs)

    with gzip.open(args.data_output, "wt") as data_out, gzip.open(args.label_output, "wt") as label_out:
        for datapoint in outputs:
            data_out.write(json.dumps(datapoint["data"]) + "\n")
            label_out.write(json.dumps(datapoint["annotations"]) + "\n")