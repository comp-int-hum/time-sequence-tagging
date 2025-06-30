import argparse
import gzip
import json
import re
from tqdm import tqdm
import logging
from utils.utility import make_parent_dirs_for_files

logger = logging.getLogger(__name__)

def filter_strings(strings, patterns):
    return [s for s in strings if not any(re.match(pattern, s) for pattern in patterns)]

def process_node(node, compiled_filters, merge_paragraphs = False, min_chapter_len = None):
    """
    Recursively process a node
    - For 'sentence' nodes: filter using sentence filters
    - For other nodes: process children recursively

    Merge single-sentence paragraphs if merge_paragraphs = True
    """
    # Filter at the base unit level
    if node.get("text", ""):
        if not any(pattern.match(node["text"]) for pattern in compiled_filters):
            return node
        else:
            return None

    # Filter out invalid nodes
    subunits = node.get("subunits", [])
    subunit_type = subunits[0].get("type") if subunits else None

    for sub in subunits:
        if sub.get("type") != subunit_type:
            logger.warning(f"Improper hierarchical level found: {sub.get('type')} incompatible with {subunit_type}")
            return None

    # Process subunits
    new_subunits = []
    for sub in node.get("subunits", []):
        processed = process_node(sub, compiled_filters, merge_paragraphs)
        if processed is not None:
            new_subunits.append(processed)

    # Handle merging at the chapter level (merge single-sentence paragraphs)
    if merge_paragraphs and node["type"] == "chapter":
        merged_subunits = []
        buffer = []  # Buffer for single sentences to merge
        
        for sub in new_subunits:
            is_single_sentence_par = (
                sub["type"] == "paragraph"
                and len(sub.get("subunits", [])) == 1
                and sub["subunits"][0]["type"] == "sentence"
            )

            if is_single_sentence_par:
                buffer.append(sub["subunits"][0])
            elif sub["type"] == "paragraph":
                if buffer:
                    sub["subunits"] = buffer + sub["subunits"]
                    buffer = []
                merged_subunits.append(sub)
            else:
                if buffer:
                    merged_subunits.append({
                        "type": "paragraph",
                        "label": None,
                        "subunits": buffer
                    })
                    buffer = []
                merged_subunits.append(sub)

        if buffer:
            if merged_subunits and merged_subunits[-1]["type"] == "paragraph":
                merged_subunits[-1]["subunits"].extend(buffer)
            else:
                merged_subunits.append({
                    "type": "paragraph",
                    "label": None,
                    "subunits": buffer
                })

        new_subunits = merged_subunits

     # Filter by minimum chapter length
    if node["type"] == "chapter" and min_chapter_len is not None:
        paragraph_count = sum(1 for sub in new_subunits if sub["type"] == "paragraph")
        if paragraph_count < min_chapter_len:
            return None

    # Remove empty nodes (for example: paragraphs with no sentences)
    if not new_subunits and node["type"] != "sentence":
        return None

    new_node = dict(node)
    new_node["subunits"] = new_subunits
    return new_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help = "Extracted data")
    parser.add_argument("--output", dest="output", help = "Output file containing extracted data")
    parser.add_argument("--filters", dest="filters", nargs = "*", default = [r"^[^a-zA-Z0-9]*[A-Z]+(?:'?[A-Z]+)?(\s[A-Z]+(?:'?[A-Z]+)?)*[^a-zA-Z0-9]*$"], help = "Sentence filter patterns")
    parser.add_argument("--min_chapter_len", dest = "min_chapter_len", type = int, help = "Minimum number of paragraphs in chapters")
    parser.add_argument("--merge_paragraphs", dest = "merge_paragraphs", action="store_true", required = False, help = "Whether to merge single sentences into other paragraphs")
    
    args = parser.parse_args()

    make_parent_dirs_for_files([args.output])
    
    compiled_filters = [re.compile(filter) for filter in args.filters]
    
    with gzip.open(args.input, "rt") as input_file, gzip.open(args.output, "wt") as output_file:
        for i, line in tqdm(enumerate(input_file), desc = "Looping over works"):
            text = json.loads(line)

            if "structure" in text:
                processed_structure = []
                for node in text["structure"]:
                    processed = process_node(node, compiled_filters, args.merge_paragraphs, args.min_chapter_len)
                    if processed is not None:
                        processed_structure.append(processed)
                
                text["structure"] = processed_structure

                if processed_structure:
                    json.dump(text, output_file)
                    output_file.write("\n")
            else:
                logger.warning(f"No 'structure' key found in {text.get('title', '[unknown title]')}")