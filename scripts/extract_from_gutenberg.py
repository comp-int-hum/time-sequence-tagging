import re
from bs4 import BeautifulSoup
from collections import OrderedDict, Counter
import argparse
import re
import os
import csv
import jsonlines
import json
from utility import make_dirs
import matplotlib.pyplot as plt
from tqdm import tqdm
import texttable as tt
import nltk

PG_TAGS = ["P", "PA", "PB", "PC", "PD", "PE", "PF", "PG", "PH", "PJ", "PK", "PL", "PM", "PN", "PQ", "PR", "PS", "PT"]

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

def is_poetry(title):
    title = title.lower()
    for word in ["poem", "poetry", "poesie", "ballad"]:
        if word in title:
            return True
    return False

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

def get_sentences(paragraph):
    return nltk.sent_tokenize(paragraph)

## ______________ END HELPER FUNCTIONS ______________________


def get_metadata_from_csv(row):
    """Get one line of metadata from Project Gutenberg csv

    Args:
        row (dict): One row in Gutenberg csv file: [Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves]
    Returns:
        tuple: A tuple containing the following:
            - has_lit_tag (bool): whether the doc is tagged as "P" (literature)
            - dict: other metadata (title, author, year, edition, pub_info, form, tags)
    """    
    locc = row["LoCC"].split(";") if row["LoCC"] else None
    has_lit_tag = any(tag[0] == "P" for tag in locc) if locc else None
    tags = [r.strip() for r in locc] if locc else None
    return has_lit_tag, {"id": row["Text#"],
                         "title":row["Title"], 
                         "author":row["Authors"], 
                         "year": row["Issued"], 
                         "edition":None, 
                         "pub_info":None, 
                         "form":None,
                         "tags":tags}


def get_href_to_anchor_map(elements):
    """Get mapping of hrefs to anchors in the Gutenberg document

    Args:
        elements (_type_): _description_

    Returns:
        dict: mapping of hrefs to anchors in document
    """
    href_to_anchor = OrderedDict()
    for ele in elements:
        anchors = ele.find_all("a")
        if anchors:
            anchor_hrefs = [a.get("href")[1:] for a in anchors]
            duplicates = set(anchor_hrefs) & set(href_to_anchor.keys())
            # If there are duplicate hrefs, then return (TODO: ensure expected behavior)
            if duplicates:
                return {}
            # Remove hrefs pertaining to images
            for i, href in enumerate(anchor_hrefs):
                if "image" not in href:
                    href_to_anchor[href] = anchors[i]
    return href_to_anchor

def process_document(soup):
    """Try to extract volumes from document using two most common text structures

    Args:
        soup (bs4.BeautifulSoup): initialized BeautifulSoup object

    Returns:
        list: list of dictionaries representing each text
    """    
    try:
        if soup.find_all("p", attrs={"class":"toc"}):
            paragraph_toc_elements = soup.find_all("p", attrs={"class":"toc"})
            href_map = get_href_to_anchor_map(paragraph_toc_elements)
            result = extract_volumes_from_ptoc(soup, href_map)
            # print(f"Top: {type(result)}")
            return result
        
        elif soup.find(attrs={"class":"chapter"}):
            tables = soup.find_all("table")
            result = extract_volumes_from_tables(soup, tables)
            # print(f"Bottom: {type(result)}")
            return result
    except Exception as e:
        print(f"Exception type: {type(e).__name__} - message: {str(e)}")
    return []



# Output: Dictionary: (volume_name, chapter_dict)
def extract_volumes_from_ptoc(soup, href_map):
    """_summary_

    Args:
        soup (bs4.BeautifulSoup): BeautifulSoup object representing parsed Gutenberg document
        href_map (dict): dictionary mapping hrefs to anchor elements

    Returns:
        _type_: _description_
    """
    # Get table of contents hrefs and anchors
    toc_hrefs = list(href_map.keys())
    toc_anchors = list(href_map.values())
    
    # Initialize vars
    curr_vol_name = ""
    volumes, chapters_list = [], []
    
    # Iterate through hrefs in document
    for i in range(len(toc_hrefs)):
        
        # Get preceding and proceeding hrefs and extract chapter using them
        first_href = toc_hrefs[i]
        second_href = toc_hrefs[i+1] if i+1 < len(toc_hrefs) else None
        chapter_paragraphs, chapter_sentences = extract_paragraphs_from_ptoc(soup, first_href, second_href)
        
        if not chapter_paragraphs: 
            # If it is not a chapter, it is a link to a new volume
            
            # Append previous volume
            if curr_vol_name and chapters_list:
                volumes.append({
                    "title": curr_vol_name,
                    "chapters": chapters_list
                })
            
            # Get new volume name
            new_vol_name = clean_string(toc_anchors[i].get_text())
            
            # Set name and chapters_list
            curr_vol_name = new_vol_name if valid_volume_header(new_vol_name) else ""
            chapters_list = []
        else:
            # If valid chapter, append it to chapters_list
            chapter_name = clean_string(toc_anchors[i].get_text())
            if "footnotes" not in chapter_name.lower():
                chapters_list.append({
                    "chapter_name": chapter_name,
                    "paragraphs": chapter_paragraphs,
                    "sentences": chapter_sentences
                })
    
    # Append final volume
    if curr_vol_name and chapters_list:
        volumes.append({
                    "title": curr_vol_name,
                    "chapters": chapters_list
                })

    return volumes

def extract_paragraphs_from_ptoc(soup, first_href, second_href):
    """Parse one chapter. A chapter is only added if there is a valid first and second anchor OR if there is no 
    second anchor (aka last chapter). Out of order chapters due to duplicates is handled by a dictionary check.

    Args:
        soup (bs4.BeautifulSoup): BeautifulSoup object representing the parsed document
        first_href (_type_): first href element
        second_href (_type_): second href element

    Returns:
        tuple:
            - list: one list of paragraphs, where each paragraph is a string
            - list: lists of sentences, where each list represents a paragraph
    """      
    start = soup.find("a", id=first_href) or soup.find("a", attrs={"name":first_href})
    end = (soup.find("a", id=second_href) or soup.find("a", attrs = {"name": second_href})) if second_href else None
    
    curr = start.find_next()
    paragraphs = []
    while curr != end:
        if not curr:
            return []
        if curr.find('a', id=second_href) or curr.find('a', attrs = {"name": second_href}):
            # Next anchor is embedded within current element
            break
        if curr.name == "p":
            par = clean_string(curr.get_text())
            if par:
                paragraphs.append(par)
        curr = curr.find_next()
    
    return (paragraphs, [get_sentences(par) for par in paragraphs])

def extract_volumes_from_tables(soup, tables):
    """Attempt to extract volumes/text from Gutenberg html using table-based structure

    Args:
        soup (bs4.BeautifulSoup): initialized BeautifulSoup object
        tables (_type_): html tables

    Returns:
        _type_: _description_
    """    
    volumes_list = []
    
    if not tables:
        return volumes_list
    
    # Iterate over all tables found in html
    for table in tables:
        
        # Check if it is a content table
        if is_content_table(table) and table.previous_sibling and table.previous_sibling.string:
            # Get href map and extract chapters
            href_map = get_href_to_anchor_map([table])
            chapters_list = extract_chapters_from_tables(soup, href_map)
            
            # Check if chapters were found
            if chapters_list:
                volume_name = clean_string(table.previous_sibling.string)
                volumes_list.append({
                    "title": volume_name,
                    "chapters": chapters_list
                })
    return volumes_list


def extract_chapters_from_tables(soup, href_map):
    """Parse chapters from Project Gutenberg html

    Args:
        soup (bs4.BeautifulSoup): initialized BeautifulSoup object
        href_map (dict): dictionary mapping hrefs to anchors

    Returns:
        list: dictionary objects representing individual chapters
    """
    toc_hrefs = list(href_map.keys())
    toc_anchors = list(href_map.values())
    chapters_list = []
    for i in range(len(toc_hrefs)):
        # Get preceding and proceeding hrefs
        first_href = toc_hrefs[i]
        second_href = toc_hrefs[i+1] if i+1 < len(toc_hrefs) else None
        
        # Extract chapters based on preceding / proceeding hrefs
        chapter_paragraphs, chapter_sentences = extract_paragraphs_from_ptoc(soup, first_href, second_href)
        
        # Check if chapter content was found
        if not chapter_paragraphs:
            # If not a chapter, handle unexpected behavior (TODO: ensure this is correct behavior)
            return None
        else:
            # If valid chapter
            chapter_name = clean_string(toc_anchors[i].get_text())
            # Ensure chapter is not related to footnotes
            if "footnotes" not in chapter_name.lower():
                chapters_list.append(
                    {
                        "chapter_name": chapter_name,
                        "paragraphs": chapter_paragraphs,
                        "sentences": chapter_sentences
                    }
                )

    return chapters_list


def parse_volumes_into_documents(file_path, metadata):
    """Given filepath and metadata, attempt to parse a Gutenberg document into volumes, splitting texts as needed
       into multiple volumes.

    Args:
        file_path (str): Gutenberg document filepath
        metadata (dict): Metadata information (title, author, year, etc.)

    Returns:
        list: list of volumes that were extracted from the document (usually one)
    """    
    if not os.path.isfile(file_path):
        print(f"File path problem with {file_path}")
        return None
        # return []
    
    processed_docs = []
    with open(file_path, "rb") as file:
        soup = BeautifulSoup(file, "html.parser", from_encoding="UTF-8")
        volumes = process_document(soup)
        for volume in volumes:
            document = metadata.copy()
            title = volume["title"]
            print(f"Title: {title}")
            # Append extracted title to metadata title for each volume
            if title.strip():
                document["title"] += " -- " + title
            
            if volume["chapters"]:
                document["chapters"] = volume["chapters"]
                processed_docs.append(document)
    return processed_docs


def process_files_from_corpus_directory(catalog_file, base_dir):
    """Process files from Gutenberg corpus directory into structured dictionaries

    Args:
        catalog_file (str): path to Project Gutenberg metadata csv
        base_dir (str): base directory to Gutenberg corpus

    Returns:
        tuple: contains the following:
               - dictionaries representing individual Gutenberg documents 
               - tag counts for the documents extracted
    """
    data = []
    tag_counts = Counter()
    potential_docs = 0
    prose = 0
    found_files = 0
    
    with open(catalog_file) as catalog:
        csv_reader = csv.DictReader(catalog)
        # Iterate through metadata csv
        for i, row in tqdm(enumerate(csv_reader)):
            
            # Extract metadata
            has_lit_tag, metadata = get_metadata_from_csv(row)
            
            # If has literature tag and a title
            if has_lit_tag and row["Title"].strip():
                potential_docs += 1
                if not is_poetry(row["Title"]):
                    prose += 1
                    
                    # Get gutenberg id and extract volumes from text
                    gutenberg_id = row["Text#"]
                    file_path = get_gb_html_dir(base_dir, gutenberg_id)
                    extracted_volumes = parse_volumes_into_documents(file_path, metadata)
                    
                    if extracted_volumes is not None:
                        found_files += 1
                    
                    # If volumes were successfully extracted
                    if extracted_volumes:
                        data.extend(extracted_volumes)
                        
                        # Count tags
                        for t in metadata["tags"]:
                            tag_counts[t] += 1                   
                        
    return data, tag_counts, ((i-1), potential_docs, prose, found_files)

# def process_files_from_local_directory(base_dir):
#     data = []
#     for filename in os.listdir(base_dir):
#         file_path = os.path.join(base_dir, filename)
#         metadata = {"title": str(file_path), "author": "author", "edition": None, "pub_info": None}
#         volume_data = extract_volumes(file_path, metadata)
#         if volume_data:
#             data.append(volume_data)
#     return data

## ______________ MAIN ____________________________________

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", dest="base_dir", help="Base directory to start searching")
    parser.add_argument("--catalog", dest="catalog_file", help="csv file")
    parser.add_argument("--output", dest="output", help="Name of output files")
    parser.add_argument("--output_catalog", dest="output_catalog", help = "Output catalog and metadata")
    parser.add_argument("--local", dest="local", nargs = "*", help="local files")
    args, rest = parser.parse_known_args()

    print(f"Begin Processing Project Gutenberg documents")

    # for output in args.outputs:
    #     make_dirs(output)

    if args.local:
        print(f"Local repo not yet configured")
        data = []
        # data = process_files_from_local_directory(args.base_dir)
    else:
        data, tag_counts, doc_counts = process_files_from_corpus_directory(args.catalog_file, args.base_dir)

    for d in data:
        assert(d != {})
    
    table_obj = tt.Texttable()
    table_obj.set_cols_width([30, 20, 10, 10])
    table_obj.set_cols_align(["l", "l", "l", "l"])
    table_obj.header(["Title", "Author", "Year", "Gutenberg ID"])
    
    unique_ids = set()
    with open(args.output, "w") as output_file:
        for doc in data:
            output_file.write(json.dumps(doc) + "\n")
            table_obj.add_row([doc["title"], doc["author"], doc["year"], doc["id"]])
            unique_ids.add(doc["id"])
            
    
    with open(args.output_catalog, "w") as output_catalog_file:
        output_catalog_file.write(f"Number of documents in pg_metadata: {doc_counts[0]} \n")
        output_catalog_file.write(f"Number of potential documents (has literature tag and valid title): {doc_counts[1]} \n")
        output_catalog_file.write(f"Number of prose documents in potential docs: {doc_counts[2]} \n")
        output_catalog_file.write(f"Number of prose documents with valid paths: {doc_counts[3]} \n")
        output_catalog_file.write(f"Number of total documents extracted: {len(data)} \n")
        output_catalog_file.write(f"Number of unique documents found: {len(unique_ids)} \n")
        output_catalog_file.write(table_obj.draw())
        
    # with open(args.output, "w") as output_file:
    #     json.dump(data, output_file)

    if tag_counts:
        tag_categories = list(tag_counts.keys())
        tag_nums = list(tag_counts.values())
        plt.bar(range(len(tag_categories)), tag_nums, align = "center")
        plt.xticks(range(len(tag_categories)), tag_categories)
        plt.savefig("gutenberg_dist_by_tags.jpg")
        plt.show()


# def parse_volumes_into_document(file_path, metadata):
#     """Given filepath and metadata, attempt to parse a Gutenberg document into volumes, splitting texts as needed
#        into multiple volumes.

#     Args:
#         file_path (str): Gutenberg document filepath
#         metadata (dict): Metadata information (title, author, year, etc.)

#     Returns:
#         list: list of volumes that were extracted from the document (usually one)
#     """    
#     if not os.path.isfile(file_path):
#         print(f"File path problem with {file_path}")
#         return []
    
#     processed_docs = []
#     with open(file_path, "rb") as file:
#         soup = BeautifulSoup(file, "html.parser", from_encoding="UTF-8")
#         volumes = process_document(soup)
#         document = {
#             **metadata,
#             "chapters": [chapter for volume in volumes for chapter in volume["chapters"]]
#         }
#         processed_docs.append(document)
#     return processed_docs