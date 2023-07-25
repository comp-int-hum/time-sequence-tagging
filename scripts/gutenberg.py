from bs4 import BeautifulSoup
from string import punctuation
import argparse
import tarfile
import json
import re
import os

# see https://github.com/comp-int-hum/gutenberg-ns-extractor/blob/045bddb8d3b264ea99a33063df5a4b3f2e7134bc/scripts/produce_sentence_corpus.py#L19-L21
def parse_gb_directory(base_dir, text_num):
    path_elements = "/".join([c for c in text_num[:-1]])
    return os.path.join(base_dir, path_elements, text_num)

def find_html_files(base_dir):
    html_files = []
    for cpath, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("h.htm"):
                html_files.append(os.path.join(cpath, file))
    return html_files

def get_metadata(soup):
    data = {}
    title_stmt = soup.find('title').string.split('by')
    data["title"] = title_stmt[0].rstrip(punctuation)
    data["author"] = title_stmt[1].strip()

    # This is just to keep the format consistent with the data gathered from Women Writers Project
    data["edition"] = None
    data["pub_info"] = None
    data["form"] = None

    return data

def get_chapter_links(soup):
    ch_links = []
    toc = soup.find_all(attrs={"class":"toc"}) or soup.find(attrs={"class":"chapter"})
    if toc:
        for a in toc:
            ch_links.extend(toc.find_all('a'))
        
    return ch_links
    
def get_chapters(soup, ch_links):
    cnum = len(ch_links)
    chapter_dict = {}
    for i in range(cnum):
        ch_start = ch_links[i]
        ch_end = ch_links[i+1] if (i+1) < cnum else None

        paragraph_dict = {}
        pnum = 0

        curr = ch_start.find_next_sibling()
        while curr != ch_end:
            if curr.name == "p":
                paragraph_dict[pnum] = curr.get_text()
                pnum += 1
            curr = curr.find_next_sibling()

        chapter_dict[ch_links[i]] = paragraph_dict
        
    return chapter_dict

    # <div class="chapter">
    # <p class="toc">



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--base_dir", dest="base_dir", help="Base directory to start searching")
    parser.add_argument("--input", dest="input", help="input html file to be processed")
    parser.add_argument("--output", dest="output", help="Name of output file")
    args, rest = parser.parse_known_args()

    with open(args.input, "r") as fp:
        soup = BeautifulSoup(fp, "html.parser")
        chapter_links = get_chapter_links(soup)
        print(chapter_links)
                

    # html_files = find_html_files(args.base_dir)

        
