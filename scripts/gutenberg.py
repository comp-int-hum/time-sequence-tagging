from bs4 import BeautifulSoup
from string import punctuation
import argparse
import re
import os
import csv
from collections import OrderedDict
import jsonlines
import json

poss_files = 0

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

# def get_metadata(soup):
#     data = {}
#     if soup.find('title') and soup.find('title').string:
#         title_stmt = soup.find('title').string.split(' by ')
#         data["title"] = title_stmt[0][:-1] if title_stmt[0][-1] == "," else title_stmt[0]
#         data["author"] = title_stmt[1].strip() if len(title_stmt) > 1 else None
#         # This is just to keep the format consistent with the data gathered from Women Writers Project
#         data["edition"] = None
#         data["pub_info"] = None
#         data["form"] = None

#     return data

def get_signifier_words():
    return ["contents", "content", "volume", "book"]

def is_volume_header(header): # get header for what is presumably a volume
    if not header:
        return False
    signifier_words = get_signifier_words()
    for word in signifier_words:
        if word in header.lower():
            return True
    return False

def is_content_table(table):
    if table.previous_sibling and table.previous_sibling.string:
        signifier_words = get_signifier_words()
        for word in signifier_words:
            if word in table.previous_sibling.string.lower():
                return True
    return False

def fill_chapter_dict_from_anchor_list(chapter_dict, anchor_list):
    hrefs = [x.get('href') for x in anchor_list]
    if anchor_list:
        duplicates = set(hrefs) & set(chapter_dict.keys())
        if duplicates:
            return
        for anchor in anchor_list:
            chapter_dict[anchor.get('href')] = anchor

def fill_volume_dict_from_table(book_volume_dict, table):
    if is_content_table(table):
        ch_links = table.find_all('a')
        if ch_links:
            book_volume_dict[table.previous_sibling.string] = ch_links # Use header of table as name of volume
# book_volume_links
    # ch_links
    # ch_links

def fill_volume_dict_from_headers(book_volume_dict, headers):
    if not headers:
        return
    for header in headers:
        if is_volume_header(header):
            next = header.find_next_sibling()
            if next:
                ch_links = next.find_all('a')
                if ch_links:
                    book_volume_dict[header.string] = ch_links

def clean_string(str):
    str = str.replace('\r', '').replace('\n', '')
    return re.sub(r'\s+', ' ', str).strip()

def get_volume_links(soup):
    book_volume_links = OrderedDict()
    global poss_files
    paragraph_toc_class = soup.find_all('p', attrs={"class":"toc"})
    if paragraph_toc_class:
        poss_files += 1
        # Assumption: only one volume for this kind of style
        ch_links = OrderedDict()
        # if is_volume_header(paragraph_toc_class[0].get_text()):
        #     print("is volume header paragraph toc")
        #     next = paragraph_toc_class[0].find_next_sibling()
        #     if next:
        #         print('got next')
        #         # print(f'next: {next}')
        #         anchor_links = next.find_all('a')
        #         # print(f"Anchor links {anchor_links}")
        #         fill_chapter_dict_from_anchor_list(ch_links, anchor_links)
        print("regular toc")
        for instance in paragraph_toc_class:
            anchor_links = instance.find_all('a')
            fill_chapter_dict_from_anchor_list(ch_links, anchor_links)
        
        book_volume_links[" "] = list(ch_links.values()) # Default behavior for one volume book
    elif soup.find(attrs={"class":"chapter"}):
        poss_files +=1
        # Frequently multi-volume texts
        tables = soup.find_all('table')
        for table in tables:
            fill_volume_dict_from_table(book_volume_links, table)
    else:
        # Usually only one volume so only get info from first toc found
        toc_div = soup.find('div', attrs={"class":"toc"})
        if toc_div:
            poss_files += 1
            headers = toc_div.find_all('h2')
            fill_volume_dict_from_headers(book_volume_links, headers)
    return book_volume_links



def get_chapters(soup, ch_list):
    
    cnum = len(ch_list)
    chapter_dict = OrderedDict()
    for i in range(cnum):
        ch_start = ch_list[i].get('href')[1:]
        ch_end = ch_list[i+1].get('href')[1:] if (i+1) < cnum else None

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
            if curr.find_all('a', attrs = {"name": ch_end}): # next anchor embedded within current element
                break
            if curr.name == "p":
                # print(curr.get_text())
                # print("\n")
                par = clean_string(curr.get_text())
                # par = curr.get_text().replace('\r', '').replace('\n', '')
                # par = re.sub(r'\s+', ' ', par).strip()
                # if par:
                if par:
                    paragraph_dict[pnum] = par
                    pnum += 1
            curr = curr.find_next()
        chapter_name = ch_list[i].string if ch_list[i].string else str(i)
        print(f"Chapter name: {chapter_name}")
        if "footnotes" not in chapter_name.lower():
            chapter_dict[clean_string(chapter_name)] = paragraph_dict

    return chapter_dict


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
    with open(args.input) as catalog:
        csv_reader = csv.DictReader(catalog)
        potential_docs = 0
        for i, row in enumerate(csv_reader):
            # For local testing
            if i > 100:
                break
            
            locc = row["LoCC"].split(";") if row["LoCC"] else None
            is_lang_lit = any(tag[0] == "P" for tag in locc) if locc else None
            if is_lang_lit and row["Title"].strip():
                potential_docs += 1
                text_num = row["Text#"]
                file_path = get_gb_html_dir(args.base_dir, text_num)
                print(f"Text number: {text_num}")
                print(f"File Path: {file_path}")
                if os.path.isfile(file_path):
                    with open(file_path, "rb") as fpointer:
                        soup = BeautifulSoup(fpointer, "html.parser", from_encoding='UTF-8')
                        result = {"title":row["Title"], "author":row["Authors"], "edition":None, "pub_info":None, "form":None}
                        volume_links = get_volume_links(soup)
                        for header, volume in volume_links.items():
                            # print(f"Header: {header}")
                            if header.strip():
                                result["title"] += " -- " + header
                            result["segments"] = get_chapters(soup, volume)
                            if result["segments"]:
                                data.append(result)
    for d in data:
        assert(d != {})

    print(f"Potential docs: {poss_files}")
    print(f"Actual docs: {len(data)}")
    with jsonlines.open(args.outputs[0], "w") as writer:
        writer.write_all(data)

    with open(args.outputs[1], "w") as output:
        json.dump(data, output)
                
        
