import argparse
import gzip
import logging
from pandas.core.ops import docstrings
from tqdm import tqdm
import sys
from collections import defaultdict
import json
from utils.utility import make_parent_dirs_for_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Todo: write test case for this
# {
#   "type": "chapter",
#   "subunits": [
#     {
#       "type": "paragraph",
#       "subunits": [
#         {"type": "sentence", "text": "A."},
#         {"type": "sentence", "text": "B."}
#       ]
#     },
#     {
#       "type": "paragraph",
#       "subunits": [
#         {"type": "sentence", "text": "C."}
#       ]
#     }
#   ]
# }

# {
#   "chapter":   [1, 0, 0],
#   "paragraph": [1, 0, 1],
# }


# All that the function needs is to know how many base unit children it has to make the determination
# first var is number of children
def extract_hierarchical_boundaries(structure):
    hierarchical_boundaries = defaultdict(list)

    def traverse(node):
        if isinstance(node, list):
            total = 0
            for child in node:
                total += traverse(child)
            return total
        elif isinstance(node, dict):
            subunits = node.get("subunits", [])
            num_descendants = 0

            if subunits and isinstance(subunits[0], dict) and "text" in subunits[0]:
                # base case
                num_descendants = len(subunits)
            else:
                # other cases
                num_descendants = traverse(subunits)
            
            # update hierarchical boundaries and return num_descendants
            node_boundaries = [1] + [0] * (num_descendants - 1)
            hierarchical_boundaries[node["type"]].extend(node_boundaries)
            return num_descendants

        else:
            logger.warning("There should not be a non-list, non-dict node in the structure, but one was found.")
            return 0
    
    traverse(structure)
    return hierarchical_boundaries

# from leaf to root
def get_hierarchy_list(structure):
    hierarchy = []

    def descend(node):
        if isinstance(node, list):
            if node:
                descend(node[0])
        elif isinstance(node, dict):
            node_type = node.get("type")
            if node_type:
                hierarchy.append(node_type)
            subunits = node.get("subunits")
            if subunits:
                descend(subunits)

    descend(structure)
    return hierarchy[::-1]  # reverse list


def get_flattened_text_units(structure):
    texts = []

    def traverse(node):
        if isinstance(node, list):
            for item in node:
                traverse(item)
        elif isinstance(node, dict):
            if "text" in node:
                texts.append(node["text"])
            elif "subunits" in node:
                traverse(node["subunits"])

    traverse(structure)
    return texts

def process_document(doc, doc_num):
    if "structure" not in doc:
        logger.warning("No structure found in document")
        return None

    structure = doc["structure"]
    flat_text = get_flattened_text_units(structure)
    hierarchy_order = get_hierarchy_list(structure)

    data_content = {
        "metadata": doc["metadata"],
        "content": flat_text,
        "key": doc_num,
        # Extra
        "structure": structure,
    }

    hierarchical_labels = {
        "base_unit": hierarchy_order[0],
        "labels": extract_hierarchical_boundaries(structure),
        "hierarchy_order": hierarchy_order,
        "source": doc["metadata"]["source"],
        "key": doc_num
    }

    return {
        "data": data_content, 
        "annotations": hierarchical_labels
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Convert nested structures to hierarchical boundary sequences")
    parser.add_argument("--input", required = True, help = "Input")
    parser.add_argument("--data_output", required = True, help = "Sequenced data output")
    parser.add_argument("--label_output", required = True, help = "Sequenced data output")
    args = parser.parse_args()

    make_parent_dirs_for_files([args.data_output, args.label_output])
    
    with gzip.open(args.input, "rt") as data_in:
        with gzip.open(args.data_output, "wt") as data_out, gzip.open(args.label_output, "wt") as label_out:
            for doc_num, doc in tqdm(enumerate(data_in), desc="converting documents"):
                doc_obj = json.loads(doc)
                result = process_document(doc_obj, doc_num)
                if result:
                    data_out.write(json.dumps(result["data"]) + "\n")
                    label_out.write(json.dumps(result["annotations"]) + "\n")