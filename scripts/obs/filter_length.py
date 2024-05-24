import argparse
import gzip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="datapath", nargs = 1, help="Primary training file")
    parser.add_argument("--output", dest="outputs", nargs=1, help="Output filepath")
    parser.add_argument("--min_sent", type=int, dest="min_sent", help="Minimum number of sentences")
    args, rest = parser.parse_known_args()
    
    with gzip.open(args.datapath) as datapath:
        for text in datapath:
            