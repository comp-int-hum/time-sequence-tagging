import argparse
import gzip
import json
import re
from tqdm import tqdm

def filter_strings(strings, patterns):
    to_ret = []
    for s in strings:
        # print(f"S: {s}", flush=True)
        if not any(pattern.match(s) for pattern in patterns):
            to_ret.append(s)
    
    return to_ret
    # return [s for s in strings if not any(re.match(pattern, s) for pattern in patterns)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help = "Extracted data")
    parser.add_argument("--output", dest="output", help = "Output file containing extracted data")
    parser.add_argument("--filters", dest="filters", nargs = "*", default = [r"^[^a-zA-Z0-9]*[A-Z]+(?:'?[A-Z]+)?(\s[A-Z]+(?:'?[A-Z]+)?)*[^a-zA-Z0-9]*$"], help = "Sentence filter patterns")
    parser.add_argument("--min_avg_chapter_len", dest = "min_avg_chapter_len", type = int, help = "Minimum number of paragraphs in chapters")
    parser.add_argument("--merge_paragraphs", dest = "merge_paragraphs", action="store_true", required = False, help = "Whether to merge single sentences into other paragraphs")
    
    args = parser.parse_args()
    
    compiled_filters = [re.compile(filter) for filter in args.filters]
    
    print(f"TEST 1")
    with gzip.open(args.input, "rt") as input_file, gzip.open(args.output, "wt") as output_file:
        for i, line in tqdm(enumerate(input_file), desc = "Looping over works"):
            text = json.loads(line)
            
            chapters = text["chapters"]
            print(f"{text['metadata']}", flush=True)
            avg_paragraph_length = sum(len(chapter) for chapter in chapters) / len(chapters) if chapters else 0
            
            if args.min_avg_chapter_len and avg_paragraph_length < args.min_avg_chapter_len:
                continue
            
            new_chapters = []

            for chapter_dict in chapters:
                chapter_name = chapter_dict["chapter_name"]
                chapter = chapter_dict["structure"]
                
                new_paragraphs = []

                for par in chapter:
                    filtered_sentences = filter_strings(par, compiled_filters)

                    if filtered_sentences:
                        new_paragraphs.append(filtered_sentences)

                if args.merge_paragraphs:
                    merged_paragraphs = []
                    buffer = []

                    for paragraph in new_paragraphs:
                        if len(paragraph) == 1:
                            buffer.extend(paragraph)
                        else:
                            if buffer:
                                paragraph = buffer + paragraph
                                buffer = []
                            merged_paragraphs.append(paragraph)

                    if buffer:
                        if merged_paragraphs:
                            merged_paragraphs[-1].extend(buffer)
                        else:
                            merged_paragraphs.append(buffer)
                            
                    new_paragraphs = merged_paragraphs

                new_chapters.append({
                    "chapter_name": chapter_name,
                    "structure": new_paragraphs
                })

            text["chapters"] = new_chapters
            json.dump(text, output_file)
            output_file.write("\n")