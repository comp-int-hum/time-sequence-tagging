import argparse
import re
import os
import gzip
import csv
import logging
import json


logger = logging.getLogger("extract_from_gutenberg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gutenberg_path", dest="gutenberg_path", help="Location of Gutenberg directory structure")
    parser.add_argument("--output", dest="output", help="Name of output file")
    parser.add_argument("--title_filters", dest="title_filters", nargs="*", default=[], help="Title patterns to exclude")
    parser.add_argument("--loc_tags_to_keep", dest="loc_tags_to_keep", default=["PS", "PE"], nargs="*", help="Acceptable Library of Congress tags")
    parser.add_argument("--must_occur", dest="must_occur", default=[], nargs="*", help="Patterns that must match the record string")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
    
    logger.info("Processing Project Gutenberg catalog")
    texts_to_use = []
    with gzip.open(os.path.join(args.gutenberg_path, "catalog.csv.gz"), "rt") as ifd:
        for row in csv.DictReader(ifd):
            tags = [tag.strip().upper() for tag in row["LoCC"].split(";")]
            if (
                    any([tag in tags for tag in args.loc_tags_to_keep]) and
                    not re.match(r"^\s*$", row["Title"], re.I|re.M|re.S) and
                    (not any([re.match(f, row["Title"], re.I|re.S|re.M) for f in args.title_filters])) and
                    all([re.match(rx, str(row), re.I|re.S|re.M) for rx in args.must_occur])
            ):
                texts_to_use.append(row)

    logger.info(
        "Found %d texts with non-empty title containing no filter patterns, the record contains all required patterns, and at least one appropriate LoC tag",
        len(texts_to_use)
    )

    logger.info("Collecting texts")
    total = 0
    with gzip.open(args.output, "wt") as ofd:
        for text in texts_to_use:
            text_num = text["Text#"]
            text_path = os.path.join(
                args.gutenberg_path,
                *[c for c in text_num[:-1]],
                text_num,
                text_num + "-h",
                text_num + "-h.htm"
            )
            if os.path.exists(text_path):
                with open(text_path, "rt", errors="ignore") as ifd:
                    total += 1
                    text["content"] = ifd.read()
                    ofd.write(json.dumps(text) + "\n")
            else:
                logger.debug("Could not find '%s' at %s", text["Title"], text_path)
    
    logger.info("Final text count based on the Gutenberg snapshot: %d", total)
