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
    data["author"] = title_stmt[1].strip() if len(title_stmt) > 1 else None

    # This is just to keep the format consistent with the data gathered from Women Writers Project
    data["edition"] = None
    data["pub_info"] = None
    data["form"] = None

    return data

def get_signifier_words():
    return ["contents", "content", "volume", "book"]

def is_volume_header(header):
    signifier_words = get_signifier_words()
    for word in signifier_words:
        if word in header:
            return True
    return False

def is_content_table(table):
    signifier_words = get_signifier_words()
    for word in signifier_words:
        if word in table.previous_sibling.string.lower():
            return True
    return False

def get_chapter_links(soup):
    # ch_links = []
    # toc = soup.find_all(attrs={"class":"toc"}) or soup.find(attrs={"class":"chapter"})
    # if toc:
    #     for a in toc:
    #         ch_links.extend(a.find_all('a'))
        
    # return ch_links
    volume_links = []
    paragraph_toc_class = soup.find_all('p', attrs={"class":"toc"})
    if paragraph_toc_class:
        ch_links = []
        for instance in paragraph_toc_class:
            links = instance.find_all('a')
            if links:
                ch_links.extend(instance.find_all('a'))
        volume_links[" "] = ch_links # Default behavior for one volume book
    elif soup.find(attrs={"class":"chapter"}):
        tables = soup.find_all('table')
        for table in tables:
            if is_content_table(table):
                ch_links = chapter_table.find_all('a')
                if ch_links:
                    volume_links[table.previous_sibling.string] = ch_links # Use header of table as name of volume
    else:
        toc_div = soup.find('div', attrs={"class":"toc"}):
        if toc_div:
            headers = toc_div.find_all('h2')
            ch_links = []
            if headers:
                for header in headers:
                    if is_volume_header(header):
                        header.find_next_sibling()
    
    return volume_links

def get_chapters(soup, ch_links):
    
    cnum = len(ch_links)
    chapter_dict = {}
    for i in range(cnum):
        ch_start = ch_links[i].get('href')[1:]
        ch_end = ch_links[i+1].get('href')[1:] if (i+1) < cnum else None

        # print(f"Start: {ch_start}")
        # print(f"ch_start href: {ch_start.get('href')}")
        # print(f"ch_end href: {ch_end.get('href')}")
        start = soup.find('a', id=ch_start) or soup.find('a', attrs = {"name":ch_start})
        # print(f"Start mod: {start}")
        end = soup.find('a', id=ch_end) or soup.find('a', attrs = {"name": ch_end}) if ch_end else None
        
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
    data = []
    with open(args.input) as catalog:
        csv_reader = csv.DictReader(catalog)
        for row in csv_reader:
            locc = row["LoCC"].split(";") if row["LoCC"] else None
            is_lang_lit = any(tag[0] == "P" for tag in locc) if locc else None
            if is_lang_lit:
                text_num = row["Text#"]
                file_path = get_gb_html_dir(args.base_dir, text_num)
                print(f"Text number: {text_num}")
                print(f"File Path: {file_path}")
                if os.path.isfile(file_path):
                    with open(file_path, "rb") as fpointer:
                        soup = BeautifulSoup(fpointer, "html.parser", from_encoding='UTF-8')
                        print(soup.original_encoding)
                        result = get_metadata(soup)
                        volume_links = get_chapter_links(soup)
                        for volume in volume_links:
                            result["segments"] = get_chapters(soup, volume)
                            data.append(result)
    
            







    # with open(args.input, "r") as fp:
    #     soup = BeautifulSoup(fp, "html.parser")
    #     result = get_metadata(soup)
    #     chapter_links = get_chapter_links(soup)
    #     result["segments"] = get_chapters(soup, chapter_links)

    with open(args.output, "w") as output:
        json.dump(data, output)
                

    # html_files = find_html_files(args.base_dir)
        
