from bs4 import BeautifulSoup
from string import punctuation
import argparse
import tarfile
import json
import re
import os
import csv

# see https://github.com/comp-int-hum/gutenberg-ns-extractor/blob/045bddb8d3b264ea99a33063df5a4b3f2e7134bc/scripts/produce_sentence_corpus.py#L19-L21
def parse_gb_directory(base_dir, text_num):
    path_elements = "/".join([c for c in text_num[:-1]])
    return os.path.join(base_dir, path_elements, text_num)

def get_gb_html_dir(base_dir, text_num):
    dir_path = parse_gb_directory(base_dir, text_num)
    return os.path.join(dir_path, text_num + "-h", text_num + "-h.htm")
    

def find_html_files(base_dir):
    html_files = []
    for cpath, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("h.htm"):
                html_files.append(os.path.join(cpath, file))
    return html_files

def get_metadata(soup):
    data = {}
    title_stmt = soup.find('title').string.split(' by ')
    data["title"] = title_stmt[0][:-1] if title_stmt[0][-1] == "," else title_stmt[0]
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
            ch_links.extend(a.find_all('a'))
        
    return ch_links

def get_chapters(soup):
    toc = soup.find_all(attrs={"class":"toc"})
    if toc:
        return get_chapters_toc(soup, toc)
    elif soup.find(attrs={"class":"chapter"}):
        return

def get_chapters_div(soup):
    chapters =
    
def get_chapters_toc(soup, toc):
    ch_links = []
    for instance in toc:
        ch_links.extend(instance.find_all('a'))
    cnum = len(ch_links)
    chapter_dict = {}
    for i in range(cnum):
        ch_start = ch_links[i]
        ch_end = ch_links[i+1] if (i+1) < cnum else None

        # print(f"ch_start href: {ch_start.get('href')}")
        # print(f"ch_end href: {ch_end.get('href')}")
        start = soup.find('a', id=ch_start.get('href')[1:])
        end = soup.find('a', id=ch_end.get('href')[1:]) if (i+1) < cnum else None
        
        paragraph_dict = {}
        pnum = 0

        curr = start.find_next()
        while curr and curr != end:
            if curr.name == "p":
                # print(curr.get_text())
                # print("\n")
                paragraph_dict[pnum] = curr.get_text()
                pnum += 1
            curr = curr.find_next()

        chapter_dict[ch_links[i].string] = paragraph_dict
        
    return chapter_dict

    # <div class="chapter">
    # <p class="toc">


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", dest="base_dir", help="Base directory to start searching")
    parser.add_argument("--input", dest="input", help="csv file")
    parser.add_argument("--output", dest="output", help="Name of output file")
    parser.add_argument("--local", dest="local", nargs="?", help="local files")
    args, rest = parser.parse_known_args()

    print(f"Input: {args.input}")
    result = {}
    with open(args.input) as catalog:
        csv_reader = csv.DictReader(catalog)
        for row in csv_reader:
            locc = row["LoCC"].split(";")
            is_lang_lit = any(tag[0] == "P" for tag in locc)
            if is_lang_lit:
                text_num = row["Text#"]
                file_path = get_gb_html_dir(args.base_dir, text_num)
                print(f"Text number: {text_num}")
                print(f"File Path: {file_path}")
                with open(file_path, "r") as fpointer:
                    soup = BeautifulSoup(fpointer, "html.parser")
                    result = get_metadata(soup)
                    ch_links = get_chapter_links(soup)
                    result["segments"] = get_chapters(soup, ch_links)
    
            







    # with open(args.input, "r") as fp:
    #     soup = BeautifulSoup(fp, "html.parser")
    #     result = get_metadata(soup)
    #     chapter_links = get_chapter_links(soup)
    #     result["segments"] = get_chapters(soup, chapter_links)

    with open(args.output, "w") as output:
        json.dump(result, output)
                

    # html_files = find_html_files(args.base_dir)
        
