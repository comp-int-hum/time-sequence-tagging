import jsonlines
import argparse
import random
import gzip
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def open_writer(path):
    return jsonlines.Writer(gzip.open(path, "wt"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input", dest="data_input", help="Data input file")
    parser.add_argument("--embedding_input", dest="embedding_input", help="Embedding input file")
    parser.add_argument("--label_input", dest="label_input", help="Label input file")

    parser.add_argument("--train", dest="train", nargs = 3, help="Names for training files")
    parser.add_argument("--dev", dest="dev", nargs = 3, help="Names for dev files")
    parser.add_argument("--test", dest="test", nargs = 3, help="Names for test files")

    parser.add_argument("--train_proportion", type=float, default=0.8)
    parser.add_argument("--dev_proportion", type=float, default=0.1)
    parser.add_argument("--test_proportion", type=float, default=0.1)

    parser.add_argument("--random_seed", dest="random_seed", type=int)
    args, rest = parser.parse_known_args()

    if args.random_seed != None:
        random.seed(args.random_seed)
        
    logger.info("Creating datapoints")

    train_writers = [open_writer(p) for p in args.train]
    dev_writers = [open_writer(p) for p in args.dev]
    test_writers = [open_writer(p) for p in args.test]

    split_counts = {"train": 0, "dev": 0, "test": 0}
    
    with gzip.open(args.data_input, "rt") as di, gzip.open(args.label_input, "rt") as li, gzip.open(args.embedding_input, "rt") as ei:
        with gzip.open(args.train, mode="wt") as train_ofd, gzip.open(args.dev, mode="wt") as dev_ofd, gzip.open(args.test, mode="wt") as test_ofd:
            counter = 0
            for idx, (data_doc, label_doc, embedding_doc) in tqdm(enumerate(zip(di, li, ei)), desc = "Iterating over datapoints"):
                
                rv = random.random()
                if rv < args.train_proportion:
                    writers = train_writers
                    split = "train"
                elif rv < args.train_proportion + args.dev_proportion:
                    writers = dev_writers
                    split = "dev"
                else:
                    writers = test_writers
                    split = "test"

                for writer, doc in zip(writers, [data_doc, label_doc, embedding_doc]):
                    writer.write(doc)

                split_counts[split] += 1

    
    for writer in train_writers + dev_writers + test_writers:
        writer.close()       

    logger.info(f"Split completed: train={split_counts['train']}, dev={split_counts['dev']}, test={split_counts['test']}")