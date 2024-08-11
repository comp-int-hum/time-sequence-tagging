import argparse
import re
import os
import gzip
import csv
import logging
import json


logger = logging.getLogger("compute_dataset_statistics")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Name of input file")
    parser.add_argument("--output", dest="output", help="Name of output file")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
    
    logger.info("Scanning dataset")
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            print(len(j["structure"][0]), j["Text#"])
