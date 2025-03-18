import os
import jsonlines
import gzip

def make_parent_dirs(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def make_parent_dirs_for_files(filepaths):
    for filepath in filepaths:
        make_parent_dirs(filepath)

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def open_file(filename, mode = "rt"):
    with open(filename, "rb") as file:
        magic_number = file.read(2)

    if magic_number == b"\x1f\x8b":
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def parse_labels(labels_str):
    label_classes = labels_str.split("=")
    return [labels.split("-") if labels else [] for labels in label_classes]

def validate_model_names(key, val, env):
    if not "sequence_tagger" or "classifier" in val:
        raise f"Invalid  model name: {val}"