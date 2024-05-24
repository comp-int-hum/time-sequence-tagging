import jsonlines
import json

# def get_gb_key(title, author):
# 	return " + ".join(title, author)

# def index_gutenberg_files(input_filepath, output_filepath):
# 	reverse_dict = {}
# 	with jsonlines.open(input_filepath, "r") as reader, open(output_filepath, "w") as out:
# 		for i, text in enumerate(reader):
# 			key = get_gb_key(text["title"], text["author"])
# 			reverse_dict[key] = i
# 		out.write(json.dump(reverse_dict))

# def get_passages(gutenberg_path, gb_index, incorrect_texts, output):
# 	with jsonlines.open(gutenberg_path, "r") as reader, open(gb_index, "r") as index, open(output, "w") as out:
# 		gb_index = json.load(index)
# 		incorrect_docs = {}
# 		for incorrect in incorrect_texts:
# 			key = get_gb_key(incorrect["title"], incorrect["author"])
# 			incorrect_docs[gb_index[key]] = (incorrect["chapters"], incorrect["first_ch"], incorrect["second_ch"])
# 		for i, text in enumerate(reader):
# 			if i in incorrect_docs:
				
def index_gutenberg_files(input_filepath, output_filepath):
	reverse_dict = {}
	with jsonlines.open(input_filepath, "r") as reader, open(output_filepath, "w") as out:
		for i, text in enumerate(reader):
			key = get_gb_key(text["title"], text["author"])
			reverse_dict[key] = i
		out.write(json.dump(reverse_dict))