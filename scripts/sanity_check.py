import argparse
import json
import jsonlines
from utility import open_file
import gzip
from tqdm import tqdm
import gzip
from collections import defaultdict, Counter
import statistics
import texttable as tt

def process_split(split_path):
    with open_file(split_path, "r") as curr_split, jsonlines.Reader(curr_split) as split_lines:
        split_counts = []
        for text in split_lines:
            counts = [text["chapter_labels"].count(0), text["chapter_labels"].count(1), 
                      text["chapter_labels"].count(2), text["paragraph_labels"].count(0), 
                      text["paragraph_labels"].count(1), text["paragraph_labels"].count(2), 
                      len(text["chapter_labels"])]
            split_counts.append(counts)
        
        # Transpose the list to get columns
        split_counts = list(zip(*split_counts))
        
        # Calculate average and stddev for each column
        averages = [statistics.mean(column) for column in split_counts]
        stddevs = [statistics.stdev(column) for column in split_counts]
        
        return averages, stddevs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", dest="splits", nargs = 3, help = "")
    parser.add_argument("--output", dest="output", help = "Metadata for splits")
    args, rest = parser.parse_known_args()
    
    # Read and cluster
            
    labels = ["chapter_labels_0", "chapter_labels_1", "chapter_labels_2", 
          "paragraph_labels_0", "paragraph_labels_1", "paragraph_labels_2", 
          "sequence_lengths"]
    
    with open(args.output, "wt") as output:
        for split in args.splits:
            averages, stddevs = process_split(split)
            output.write(f"Split: {split} \n")
            
            table = tt.Texttable()
            table.set_cols_align(["l", "r", "r"])
            table.add_row(["Stat/Label", "Average", "Standard Deviation"])
            
            for label, avg, std in zip(labels, averages, stddevs):
                table.add_row([label, f"{avg:.2f}", f"{std:.2f}"])
            
            output.write(table.draw() + "\n\n")
            output.write("\n")

            






        
    
    #     print(f"SANITY CHECK PLEASE SANITY")
    # for split in args.splits:
    #     with open_file(split, "r") as curr_split, jsonlines.Reader(curr_split) as split_lines:
    #         split_counts = []
    #         for text in split_lines:
    #             counts = [text["chapter_labels"].count(0), text["chapter_labels"].count(1), text["chapter_labels"].count(2),
    #                       text["paragraph_labels"].count(0), text["paragraph_labels"].count(1), text["paragraph_labels"].count(2), 
    #                       len(text["chapter_labels"])]
    #             split_counts.append(counts)
                
    #         zip(*split_counts)
    


                    
    
                
