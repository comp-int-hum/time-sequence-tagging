import gzip
import jsonlines

def count_lines_in_gzip_jsonlines(file_path):
    count = 0
    with gzip.open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            count += chunk.count(b'\n')
            print(count)
    return count



if __name__ == "__main__":
	with open("./work/data/gutenberg.jsonl", mode="rt") as gzfile:
		reader = jsonlines.Reader(gzfile)
		i = 0
		for l in reader:
			i += 1
			print(i)
                  
	# file_path = "./work/data/pg_encoded.jsonl.gz"
	# number_of_lines = count_lines_in_gzip_jsonlines(file_path)
	# print(f"The file has {number_of_lines} lines.")
