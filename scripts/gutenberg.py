from bs4 import BeautifulSoup
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

def process(soup):
        

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument("--base_dir", dest="base_dir", help="Base directory to start searching")
        parser.add_argument("--input", dest="input", help="input html file to be processed")
	parser.add_argument("--output", dest="output", help="Name of output file")
        
	args, rest = parser.parse_known_args()

        with open(args.input, "r") as fp:
                soup = BeautifulSoup(fp, "xml")
                

	# html_files = find_html_files(args.base_dir)

        
