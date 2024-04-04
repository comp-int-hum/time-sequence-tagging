import argparse
import json
import jsonlines
from encode_data import get_paragraph_sentences

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

# Input: chapter_dict representing one chapter: key=paragraph_num, value = string paragraph_content
# Output: list of all sentences in the chapter
def get_chapter_sentences(chapter_dict):
    all_sent = []
    paragraph_sentences = get_paragraph_sentences(chapter_dict) # dictionary
    for sentences in paragraph_sentences.values():
        all_sent.extend(sentences)
    return all_sent

def get_passage(pg_text, incorrect_text):
    print(f"Incorrect text: {incorrect_text.keys()}")
    ch_names = incorrect_text["chapters"]
    
    first_start, first_end = incorrect_text["first_ch"]
    
    if not ch_names[0] in pg_text["segments"]:
        return ""
    first_chapter = get_chapter_sentences(pg_text["segments"][ch_names[0]])
    first_passage = first_chapter[first_start : first_end]

    if len(ch_names) == 2:
        if not ch_names[1] in pg_text["segments"]:
            return ""
        second_chapter = get_chapter_sentences(pg_text["segments"][ch_names[1]])
        second_start, second_end = incorrect_text["second_ch"]
        second_passage = second_chapter[second_start : second_end]
        return {"Title": pg_text["title"], "Author": pg_text["author"], "POSITIVE" : " [SEP] ".join(first_passage) + " **** CHAPTER BOUNDARY **** " + " [SEP] ".join(second_passage)}
    
    return {"Title": pg_text["title"], "Author": pg_text["author"], "NEGATIVE": " [SEP] ".join(first_passage)}
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input files")
    parser.add_argument("--pg_path", dest="pg_path", help="Project Gutenberg path")
    parser.add_argument("--output", dest="output", help="Output files")
    # parser.add_argument("--context_size", dest="context", nargs=1, help="context size of original exp")
    args, rest = parser.parse_known_args()
    
    print(args.pg_path)
    print(type(args.pg_path))
    reverse_dict = get_reverse_dict(args.pg_path)
    incorrect_texts = []
    with open(args.input, "r", encoding="utf-8") as input:
        # json_next = False
        # for line in input:
        #     if json_next:
        #         print(line)
        #         incorrect_texts = json.loads(line)
        #         json_next = False
        #     elif "***" in line:
        #         json_next = True
        incorrect_texts = json.load(input)
    print(f"Len of reverse: {len(reverse_dict)}")
    line_nums = get_lines_to_read(reverse_dict, incorrect_texts)
    print(f"Line nnums: {line_nums}")
    pg_texts = get_texts(args.pg_path, line_nums)
    print(pg_texts)
    with open(args.output, "w") as output:
        for incorrect in incorrect_texts:
            output.write(json.dumps(get_passage(pg_texts[incorrect["id"]], incorrect)) + "\n")


                    
    
                
