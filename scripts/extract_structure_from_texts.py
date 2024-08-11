import sys
import re
from bs4 import BeautifulSoup
from collections import OrderedDict, Counter
import argparse
import re
import os
import gzip
import csv
import logging
import json
import nltk


logger = logging.getLogger("extract_structure_from_texts")


toc_words = ["contents", "content", "volume", "book"]

not_toc_words = ["footnotes", "index"]

def get_volumes(soup, record):
    # pretend like there are no volumes/anthologies/etc
    return [soup]
    # retval = []
    # paragraph_toc_elements = set([p.parent for p in soup.find_all("p", attrs={"class":"toc"})])
    # if paragraph_toc_elements:
    #     retval =  paragraph_toc_elements
    # else:
    #     toc_tables = [
    #         tbl.previous_sibling for tbl in soup.find_all(["p", "table"]) if
    #         tbl.previous_sibling and
    #         (
    #             (
    #                 tbl.previous_sibling.string and
    #                 any([w in tbl.previous_sibling.string.lower() for w in toc_words])
    #             )
    #             or
    #             (
    #                 tbl.previous_sibling.previous_sibling and
    #                 tbl.previous_sibling.previous_sibling.string and
    #                 any([w in tbl.previous_sibling.previous_sibling.string.lower() for w in toc_words])
    #             )
    #         )
    #     ]
    #     if not toc_tables:
    #         print(record["Title"])
    #     print(record["content"])
    #     retval = toc_tables if toc_tables else [soup]
    # return retval

def get_chapters(volume):
    # if there are straightforward chapter divs, use those, otherwise collect p-elements between a-elements that have name attributes
    chs = soup.find_all("div", attrs={"class" : "chapter"})
    if chs:
        return chs
    else:
        chapters = []
        cur_p_list = []
        for node in soup.find_all(["a", "p"]):
            if node.name == "a":
                if "name" in node.attrs and len(cur_p_list) > 0:
                    chapters.append(cur_p_list)
                    cur_p_list = []
            elif node.name == "p" and not node.find("p"):
                cur_p_list.append(node)                
        if len(cur_p_list) > 0:
            chapters.append(cur_p_list)
        return chapters

def get_paragraphs(chapter):
    return chapter if isinstance(chapter, list) else [p for p in chapter.find_all("p")]

def get_sentences(paragraph):
    return nltk.sent_tokenize(re.sub(r"\s+", " ", " ".join(paragraph.strings)))

def get_structure(soup, record):
    return [[[get_sentences(par) for par in get_paragraphs(chap)] for chap in get_chapters(vol)] for vol in get_volumes(soup, record)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Name of input file")
    parser.add_argument("--output", dest="output", help="Name of output file")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
    
    logger.info("Extracting structure")

    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            soup = BeautifulSoup(j["content"], "html.parser")
            nj = {k : v for k, v in j.items() if k != "content"}
            nj["structure"] = get_structure(soup, j)
            ofd.write(json.dumps(nj) + "\n")
