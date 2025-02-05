import argparse
from tqdm import tqdm
import gzip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest = "input", help = "Input file")
    parser.add_argument("--output", dest = "output", help = "Output file containing truncated inputs")
    parser.add_argument("--retain_lines", dest = "retain_lines", type = int, help = "Number of lines to retain")
    args, _ = parser.parse_known_args()
    
    with gzip.open(args.input, "rt") as input_file, gzip.open(args.output, "wt") as output_file:
        for _ in tqdm(range(args.retain_lines), desc = "Downsampling files"):
            next_text = next(input_file, None)
            if next_text is not None:
                output_file.write(next_text)
            else:
                break