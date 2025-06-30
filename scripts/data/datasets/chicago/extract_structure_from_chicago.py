import zipfile
import argparse
from bs4 import BeautifulSoup
import re
import nltk
import gzip
import json
import logging
from tqdm import tqdm
import texttable as tt

logger = logging.getLogger(__name__)

non_toc_words = ["footnotes", "biographical note", "copyright"]

def get_chapters(soup, max_title_len, chapter_heading_patterns, non_toc_words):
    chapters = []
    chapter_names = []
    curr_p_list = []
    first_match = False

    # Precompiled filters
    chapter_regexes = [re.compile(pattern, re.I) for pattern in chapter_heading_patterns]
    non_toc_regexes = [re.compile(pattern, re.I) for pattern in non_toc_words]

    for node in soup.find_all("p"):
        text = node.get_text().strip()
        
        if all(ch_regex.match(text) for ch_regex in chapter_regexes):
            logger.debug(f"Matched: {text}")
            chapter_names.append(text)
            
            if curr_p_list:
                chapters.append(curr_p_list)
                curr_p_list = []
                
            first_match = True
        elif len(text) < max_title_len and any(toc_regex.match(text) for toc_regex in non_toc_regexes):
            logger.debug(f"Skipped/broke on: {text}")
            break 

        elif first_match:
            curr_p_list.append(node)
            
    if first_match and curr_p_list:
        chapters.append(curr_p_list)

    return chapter_names, chapters

def get_sentences(paragraph):
    if isinstance(paragraph, list):
        text = " ".join(node.get_text(strip=True) for node in paragraph)
    else:
        text = paragraph.get_text(strip=True)
    return nltk.sent_tokenize(text)

def get_metadata(soup, text_id=None, segment_num=None, source=None):
    def clean_text(tag):
        return tag.string.strip() if tag and tag.string else None

    return {
        "title": clean_text(soup.find("title")),
        "author": clean_text(soup.find("author").persName) if soup.find("author") else None,
        "text_id": text_id,
        "segment_num": segment_num,
        "year": clean_text(soup.find("publicationStmt").date) if soup.find("publicationStmt") else None,
        "source": source,
    }
    
def get_structure(soup, max_title_len, chapter_heading_patterns):
    chapters = []
    for chapter_name, chapter in zip(*get_chapters(soup, max_title_len, chapter_heading_patterns, non_toc_words)):
        chapters.append({
            "type": "chapter",
            "label": chapter_name,
            "subunits": [
                {
                    "type": "paragraph",
                    "label": None,
                    "subunits": [
                        {
                            "type": "sentence",
                            "label": None,
                            "text": sentence
                        }
                        for sentence in get_sentences(par)
                    ]
                }
                for par in chapter
            ]
        })
    return chapters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chicago_path", dest="chicago_path", help = "Location of zipfile for Chicago pre-1927 novels")
    parser.add_argument("--output", dest="output", help = "Output file containing extracted data")
    parser.add_argument("--output_catalog", dest="output_catalog", help = "Metadata of extracted files")
    parser.add_argument("--min_chapters", dest="min_chapters", type=int, default=3)
    parser.add_argument("--max_title_len", dest="max_title_len", type=int, default=500)
    parser.add_argument("--chapter_heading_patterns", dest="chapter_filters", nargs = "*", default = [], help = "Chapter title filter patterns")
    args = parser.parse_args()
    
    # Initialize table
    partial_extracted = 0
    fully_extracted = 0
    total = 0
    
    catalog_table_obj = tt.Texttable()
    catalog_table_obj.set_cols_width([30, 20, 10])
    catalog_table_obj.set_cols_align(["l", "l", "l"])
    catalog_table_obj.header(["Title", "Author", "Year"])

    # Check file and extract
    
    if zipfile.is_zipfile(args.chicago_path):
        with zipfile.ZipFile(args.chicago_path) as zf, gzip.open(args.output, "wt") as ofd:
            for zip_info in tqdm(zf.infolist()):
                filename = zip_info.filename
                text_id_match = re.search(r"clean(\d+)\.tei", filename)
                text_id = text_id_match.group(1) if text_id_match else None

                total += 1
                with zf.open(zip_info, mode="r") as text:
                    soup = BeautifulSoup(text, "xml")

                    logger.debug(f"--- {zip_info.filename} ---")

                    extracted_text = {
                        "metadata": get_metadata(
                            soup,
                            text_id = text_id,
                            segment_num = 0,
                            source = "chicago_pre1927"
                        ),
                        "structure": get_structure(soup, args.max_title_len, args.chapter_filters)
                    }
                    
                    if extracted_text["chapters"]:
                        partial_extracted += 1
                        if len(extracted_text["chapters"]) > args.min_chapters:
                            fully_extracted += 1
                            catalog_table_obj.add_row(list(extracted_text["metadata"].values()))
                            ofd.write(json.dumps(extracted_text) + "\n")
                        else:
                            logger.debug(f"Partially extracted {len(extracted_text['chapters'])} chapters from {zip_info.filename}")
                    else:
                        logger.debug(f"Could not extract {zip_info.filename}")
    
    # Table
    stats_table = tt.Texttable()
    stats_table.set_cols_width([15, 15, 15])
    stats_table.set_cols_align(["l", "l", "l"])
    stats_table.header(["Partially Extracted", "Fully Extracted", "Total"])
    stats_table.add_row([partial_extracted, fully_extracted, total])  

    logger.info(f"Partially extracted texts: {partial_extracted}")
    logger.info(f"Fully extracted: {fully_extracted}")    
    logger.info(f"Total number of texts: {total}")


    # Output    
    with gzip.open(args.output_catalog, "wt") as output_catalog_file:
        output_catalog_file.write(catalog_table_obj.draw() + "\n\n")
        output_catalog_file.write(stats_table.draw())