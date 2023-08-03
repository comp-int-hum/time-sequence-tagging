from bs4 import BeautifulSoup
from string import punctuation
import argparse
import re
import os
import csv
import jsonlines

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
    signifier_words = get_signifier_words()
    for word in signifier_words:
        if word in header:
            return True
    return False

def is_content_table(table):
    if table.previous_sibling and table.previous_sibling.string:
        signifier_words = get_signifier_words()
        for word in signifier_words:
            if word in table.previous_sibling.string.lower():
                return True
    return False

def get_volume_links(soup):
    volume_links = {}
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
                ch_links = table.find_all('a')
                if ch_links:
                    volume_links[table.previous_sibling.string] = ch_links # Use header of table as name of volume
    else:
        toc_div = soup.find('div', attrs={"class":"toc"})
        if toc_div:
            headers = toc_div.find_all('h2')
            if headers:
                for header in headers:
                    ch_links = []
                    if is_volume_header(header):
                        next = header.find_next_sibling()
                        if next:
                            ch_links = next.find_all('a')
                            if ch_links:
                                volume_links[header.string] = ch_links
    
    return volume_links



def get_chapters(soup, ch_links):
    
    cnum = len(ch_links)
    chapter_dict = {}
    try:
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
                    par = curr.get_text().replace('\r', '').replace('\n', '')
                    par = re.sub(r'\s+', ' ', par).strip()
                    if par:
                        paragraph_dict[pnum] = par
                        pnum += 1
                curr = curr.find_next()

            chapter_dict[ch_links[i].string] = paragraph_dict
    except:
        return None
    return chapter_dict


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
        for i, row in enumerate(csv_reader):
            # For local testing
            if i > 100:
                break
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
                        print(soup.original_encoding)
                        result = {"title":row["Title"], "author":row["Authors"], "edition":None, "pub_info":None, "form":None}
                        volume_links = get_volume_links(soup)
                        for header, volume in volume_links.items():
                            print(f"Header: {header}")
                            if header.strip():
                                result["title"] += " -- " + header
                            result["segments"] = get_chapters(soup, volume)
                            if result["segments"]:
                                data.append(result)

    with open(args.output, "w") as output:
        jsonlines.write_all(data, output)
                
        
