import re
from bs4 import BeautifulSoup
from collections import OrderedDict
import argparse
import re
import os
import csv
import jsonlines
import json

potential_docs = 0
## ______________ HELPER FUNCTIONS ______________________

# see https://github.com/comp-int-hum/gutenberg-ns-extractor/blob/045bddb8d3b264ea99a33063df5a4b3f2e7134bc/scripts/produce_sentence_corpus.py#L19-L21
def parse_gb_directory(base_dir, text_num):
    path_elements = "/".join([c for c in text_num[:-1]])
    return os.path.join(base_dir, path_elements, text_num)

def get_gb_html_dir(base_dir, text_num):
    dir_path = parse_gb_directory(base_dir, text_num)
    return os.path.join(dir_path, text_num + "-h", text_num + "-h.htm")

def clean_string(str):
    str = str.replace('\r', '').replace('\n', '')
    return re.sub(r'\s+', ' ', str).strip()

def is_content_table(table):
    if table.previous_sibling and table.previous_sibling.string:
        signifier_words = get_signifier_words()
        for word in signifier_words:
            if word in table.previous_sibling.string.lower():
                return True
    return False

def get_signifier_words():
    return ["contents", "content", "volume", "book"]

def get_invalid_words():
    return ["footnotes", "index"]

def contains_invalid_words(string):
    invalid_words = get_invalid_words()
    for invalid in invalid_words:
        if invalid in string.lower():
            return True
    return False

def valid_volume_header(string):
    return string.strip() and not contains_invalid_words(string)
## ______________ END HELPER FUNCTIONS ______________________


# Input: first anchor, second anchor
# If chapter is invalid, return None.
# If chapter is valid, return text in chapter
# A chapter is only added if there is a valid first anchor and a valid second anchor OR if there is no second anchor (aka last chapter)
# Out of order chapters due to duplicates is handled by a dictionary check
def get_chapter(soup, first, second):   
    start = soup.find('a', id=first) or soup.find('a', attrs={"name":first})
    end = (soup.find('a', id=second) or soup.find('a', attrs = {"name": second})) if second else None
    
    curr = start.find_next()
    paragraph_dict = {}
    pnum = 0
    while curr != end:
        if not curr:
            return None
        if curr.find('a', id=second) or curr.find('a', attrs = {"name": second}):
            # next anchor embedded within current element
            break
        if curr.name == "p":
            par = clean_string(curr.get_text())
            if par:
                paragraph_dict[pnum] = par
                pnum += 1
        curr = curr.find_next()
    
    if paragraph_dict:
        return paragraph_dict
    return None

# Input: list of elements containing anchors
# Output: dict of non-duplicate links
def get_links(elements):
    links = OrderedDict()
    for ele in elements:
        anchors = ele.find_all('a')
        if anchors:
            anchor_hrefs = [x.get('href')[1:] for x in anchors]
            duplicates = set(anchor_hrefs) & set(links.keys())
            # if duplicate anchors, break
            if duplicates:
                break
            for anchor in anchors:
                if "image" not in anchor.get('href')[1:]:
                    links[anchor.get('href')[1:]] = anchor
    return links

# Input: soup object, OrderedDict of links (href, anchor element)
# Output: Dictionary: (volume_name, chapter_dict)
def get_volumes_ptoc(soup, links):
    toc_hrefs = list(links.keys())
    toc_anchors = list(links.values())
    volumes = {}
    curr_vol_name = ""
    chapter_dict = OrderedDict()
    for i in range(len(toc_hrefs)):
        first_href = toc_hrefs[i]
        second_href = toc_hrefs[i+1] if i+1 < len(toc_hrefs) else None
        chapter_content = get_chapter(soup, first_href, second_href)
        if not chapter_content: # if not a chapter, just a link
            volumes[curr_vol_name] = chapter_dict
            # if valid_volume_header(curr_vol_name): # if not empty and not footnote
            #     volumes[curr_vol_name] = chapter_dict # add previous volume
            #     print(f"TOC anchor: {curr_vol_name}")
            new_vol_name = clean_string(toc_anchors[i].get_text()) # update to new vol_name
            if valid_volume_header(new_vol_name):
                curr_vol_name = new_vol_name
                chapter_dict = OrderedDict() # new empty dict
            print(f"New volume name: {curr_vol_name}")
        else: # if valid chapter
            chapter_name = clean_string(toc_anchors[i].get_text())
            if "footnotes" not in chapter_name.lower():
                print(f"Chapter name: {chapter_name}")
                chapter_dict[chapter_name] = chapter_content

    volumes[curr_vol_name] = chapter_dict
    # print(chapter_dict)
    if not volumes[""]:
        volumes.pop("")
    # print(volumes)
    return volumes

# Input: links for one volume
# Output: dictionary (chapter_name, paragraph_dict); could be empty if it finds an invalid chapter
def get_chapters(soup, links):
    toc_hrefs = list(links.keys())
    toc_anchors = list(links.values())
    chapter_dict = OrderedDict()
    for i in range(len(toc_hrefs)):
        first_href = toc_hrefs[i]
        second_href = toc_hrefs[i+1] if i+1 < len(toc_hrefs) else None
        chapter_content = get_chapter(soup, first_href, second_href)
        if not chapter_content: # if not a chapter --> unexpected behavior
            break
        else: # if valid chapter
            chapter_name = clean_string(toc_anchors[i].get_text())
            if "footnotes" not in chapter_name.lower():
                chapter_dict[chapter_name] = chapter_content

    return chapter_dict

def get_volumes_tables(soup, tables):
    volumes = {}
    if not tables:
        return volumes
    
    for table in tables:
        if is_content_table(table) and table.previous_sibling and table.previous_sibling.string:
            links = get_links([table])
            chapter_dict = get_chapters(soup, links)
            if chapter_dict:
                volume_name = clean_string(table.previous_sibling.string)
                volumes[volume_name] = chapter_dict
    return volumes

# Get volumes
def get_volumes(soup):
    global potential_docs
    volumes = {}
    try:
        if soup.find_all('p', attrs={"class":"toc"}):
            potential_docs += 1
            paragraph_toc_elements = soup.find_all('p', attrs={"class":"toc"})
            links = get_links(paragraph_toc_elements)
            volumes = get_volumes_ptoc(soup, links)
            # print(f"TOC Volumes: {volumes}")
            return volumes
        
        elif soup.find(attrs={"class":"chapter"}):
            potential_docs += 1
            tables = soup.find_all('table')
            volumes = get_volumes_tables(soup, tables)
            # print(f"Chapter Volumes: {volumes}")
            return volumes
    except:
        return volumes
    return volumes
    

## ______________ MAIN ____________________________________

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", dest="base_dir", help="Base directory to start searching")
    parser.add_argument("--input", dest="input", help="csv file")
    parser.add_argument("--output", dest="outputs", nargs="*", help="Name of output files")
    parser.add_argument("--local", dest="local", nargs="?", help="local files")
    args, rest = parser.parse_known_args()

    print(f"Outputs: {args.outputs}")
    print(f"Input: {args.input}")

    data = []
    if args.local:
        files = os.listdir(args.base_dir)
        for filename in files:
            file_path = os.path.join(args.base_dir, filename)
            if os.path.isfile(file_path):
                print(f"File Path: {file_path}")
                with open(file_path, 'r') as file:
                    soup = BeautifulSoup(file, "html.parser", from_encoding="UTF-8")
                    volumes = get_volumes(soup)
                    # print(volumes.items())
                    print(volumes.keys())
                    metadata = {"title": str(file)}
                    for header, chapters in volumes.items():
                        result = metadata.copy()
                        if header.strip():
                            result["title"] += " -- " + header
                        result["segments"] = chapters
                        if result["segments"]:
                            data.append(result)
    else:
        with open(args.input) as catalog:
            csv_reader = csv.DictReader(catalog)
            potential_docs = 0
            for i, row in enumerate(csv_reader):
                # For local testing
                # if i > 100:
                #     break

                # if i != x:
                #   continue
                
                locc = row["LoCC"].split(";") if row["LoCC"] else None
                is_lang_lit = any(tag[0] == "P" for tag in locc) if locc else None
                if is_lang_lit and row["Title"].strip():
                    text_num = row["Text#"]
                    file_path = get_gb_html_dir(args.base_dir, text_num)
                    print(f"Text number: {text_num}")
                    print(f"File Path: {file_path}")
                    if os.path.isfile(file_path):
                        with open(file_path, "rb") as fpointer:
                            soup = BeautifulSoup(fpointer, "html.parser", from_encoding='UTF-8')
                            metadata = {"title":row["Title"], "author":row["Authors"], "edition":None, "pub_info":None, "form":None}
                            volumes = get_volumes(soup)
                            for header, chapters in volumes.items():
                                result = metadata.copy()
                                if header.strip():
                                    result["title"] += " -- " + header
                                result["segments"] = chapters
                                if result["segments"]:
                                    data.append(result)
    for d in data:
        assert(d != {})

    print(f"Potential docs: {potential_docs}")
    print(f"Actual docs: {len(data)}")
    with jsonlines.open(args.outputs[0], "w") as writer:
        writer.write_all(data)

    with open(args.outputs[1], "w") as output:
        json.dump(data, output)


# if __name__ == "__main__":
#     print(contains_invalid_words("CHAPTER I"))

# BeautifulSoup functionality testing
# if __name__ == "__main__":
#     test_str = '<div> <p> Some text <a id="href_end">Anchor 1</a></p> <p>Other text <a id="another_id">Anchor 2</a></p> </div>'
#     soup = BeautifulSoup(test_str, "html.parser")

#     curr = soup.find('p')
#     while curr:
#         print(f"Curr: {curr}")
#         curr = curr.find_next()
