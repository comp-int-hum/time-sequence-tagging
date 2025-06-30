import re
from bs4 import BeautifulSoup
import argparse
import re
import gzip
import csv
import logging
import json
import nltk

logger = logging.getLogger(__name__)


toc_words = ["contents", "content", "volume", "book"]

not_toc_words = ["footnotes", "index"]

# def get_volumes(soup, record):
#     # pretend like there are no volumes/anthologies/etc
#     return [soup]

def get_chapters(soup):
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

def get_structure(soup):
    return  [   
                {
                    "type": "chapter",
                    "label": None,
                    "subunits": [
                            {
                                "type": "paragraph",
                                "label": None,
                                "subunits": [
                                    {
                                        "type": "sentence",
                                        "label": None,
                                        "text": sent
                                    }
                                    for sent in get_sentences(par)
                                ]
                            }
                            for par in get_paragraphs(chap)
                    ]
                } 
                for chap in get_chapters(soup)
            ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Name of input file")
    parser.add_argument("--output", dest="output", help="Name of output file")
    parser.add_argument("--limit", dest = "limit", type = int, required = False, help = "Structure extraction limit")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
    
    logger.info("Extracting structure")

    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, line in enumerate(ifd):
            if args.limit and i >= args.limit:
                break
            j = json.loads(line)
            soup = BeautifulSoup(j["content"], "html.parser")
            nj = {k : v for k, v in j.items() if k != "content"}
            nj["structure"] = get_structure(soup)
            ofd.write(json.dumps(nj) + "\n")