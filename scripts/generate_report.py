import argparse
import json
import jsonlines
from encode_data import get_paragraphs

#
# This script does *nothing* except print out its arguments and touch any files
# specified as outputs (thus fulfilling a build system's requirements for
# success).
#
def get_reverse_dict(pg_path):
    reverse_dict = {}
    with jsonlines.open(pg_path, "r") as reader:
        for i, text in enumerate(reader):
            reverse_dict[text["id"]] = i
    return reverse_dict

def get_lines_to_read(reverse_dict, texts):
    line_nums = []
    for text in texts:
        line_nums.append(reverse_dict[text["id"]])
    return line_nums

def get_texts(pg_path, line_nums):
    texts = {}
    with jsonlines.open(pg_path, "r") as reader:
        for i, text in enumerate(reader):
            if i in line_nums:
                texts[text["id"]] = text
    return texts

def get_error_output(pg_text, model_type, incorrect_text):
    output = {"Title": pg_text["title"], "Author": pg_text["author"]}
    if model_type == "classifier":
        if incorrect_text["ground_truth"]:
            output["POSITIVE"] = " ".join(incorrect_text["first_ch"]) + "****** CHAPTER BOUNDARY ******" + " ".join(incorrect_text["second_ch"]) 
        else:
            output["NEGATIVE"] = " ".join(incorrect_text["first_ch"])
    else:
        output["LABELS"] = incorrect_text["labels"]
        # output["ACCURACY"] = incorrect_text["accuracy"]
        # output["PRECISION"] = incorrect_text["precision"]
        # output["RECALL"] = incorrect_text["recall"]
        output["ERRORS"] = incorrect_text["errors"]
    return output
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input files")
    parser.add_argument("--pg_path", dest="pg_path", help="Project Gutenberg path")
    parser.add_argument("--model_type", dest="model_type", help="Classifier vs sequence tagger")
    parser.add_argument("--output", dest="output", help="Output files")
    # parser.add_argument("--context_size", dest="context", nargs=1, help="context size of original exp")
    args, rest = parser.parse_known_args()
    
    print(args.pg_path)
    print(type(args.pg_path))
    reverse_dict = get_reverse_dict(args.pg_path)
    incorrect_texts = []
    with open(args.input, "r", encoding="utf-8") as input:
        incorrect_texts = json.load(input)
    print(f"Len of reverse: {len(reverse_dict)}")
    line_nums = get_lines_to_read(reverse_dict, incorrect_texts)
    print(f"Line nums: {line_nums}")
    pg_texts = get_texts(args.pg_path, line_nums)
    with open(args.output, "w") as output:
        i = 0
        for incorrect in incorrect_texts:
            i+= 1
            output.write(json.dumps(get_error_output(pg_texts[incorrect["id"]], args.model_type, incorrect)) + "\n")
        print(f"NUM INCORRECT TEXTS: {i}")


                    
    
                
