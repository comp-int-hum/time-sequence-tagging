import argparse
import torch
import gzip
from tqdm import tqdm
import json
import nltk
import gzip
from utility import make_dirs
import re
import random

def filter_strings(strings, patterns):
    to_ret = []
    for s in strings:
        if not any(pattern.match(s) for pattern in patterns):
            to_ret.append(s)
    
    return to_ret

def get_hierarchical_labels(context_len, pos_len):
    hierarchical_labels = [[0] for _ in range(context_len + pos_len)]
    hierarchical_labels[context_len] = [1]
    hierarchical_labels[context_len - 1] = [1]
    return hierarchical_labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Chapterbreak file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--splits", dest = "splits", nargs = "+", default = ["pg19", "ao3"])
    parser.add_argument("--filters", dest="filters", nargs = "*", default = [r"^[^a-zA-Z0-9]*[A-Z]+(?:'?[A-Z]+)?(\s[A-Z]+(?:'?[A-Z]+)?)*[^a-zA-Z0-9]*$"], help = "Sentence filter patterns")
    args, rest = parser.parse_known_args()
    
    make_dirs(args.output)
    
    with open(args.input, "rt") as input_file:
        data = json.load(input_file)
    
    compiled_filters = [re.compile(filter) for filter in args.filters]
    
    output_data = []
    for split_name in args.splits:
        data_split = data[split_name]
        for text_id, text in tqdm(list(data_split.items()), desc = f"Iterating over texts in split: {split_name}"):
            for idx, triplet in enumerate(text):
                context = triplet["ctx"]
                pos = triplet["pos"]
                context_sentences = filter_strings(nltk.sent_tokenize(context), compiled_filters)
                pos_sentences = filter_strings(nltk.sent_tokenize(pos), compiled_filters)
                output_data.append(
					{
						"metadata": {
							"title": text_id,
							"author": "",
							"text_id": text_id,
							"segment_num": idx
						},
						"flattened_sentences": context_sentences + pos_sentences,
						"hierarchical_labels": get_hierarchical_labels(len(context_sentences), len(pos_sentences)),
					}
				)
    random.shuffle(output_data)
    
    with gzip.open(args.output, "wt") as output_file:
        for chapter_break in output_data:
        	output_file.write(json.dumps(chapter_break) + "\n")
    
                