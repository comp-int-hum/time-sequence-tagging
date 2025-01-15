import argparse
import jsonlines 
import random
import gzip
from utility import open_file
from tqdm import tqdm

# MIN_PAR_LENGTH = 2

QUOTE_PUNCTS = [
    # ASCII
    '"', "'", "``", "’",     

    # Smart quotes
    "“", "”", "‘", "’",     

    # Angle
    "«", "»", "‹", "›",     

    # Brackets
    "「", "」", "『", "』",    

    # Other
    "„", "”", "‚"      
]

# def get_ibe_tag(ele_num, seq_len):
#     """Get inside, beginning, ending tag for a paragraph or chapter.

#     Args:
#         num (int): The index of the element in the sequence
#         seq_len (int): The total length of the sequence

#     Returns:
#         int: (0) for inside, (1) for beginning / boundary, (2) for ending
#     """    
#     tag = 0
#     # if (seq_len == 1):
#     #     tag = 3
#     if ele_num == 0:
#         tag = 1
#     elif ele_num == (seq_len - 1):
#         tag = 2
#     else:
#         tag = 0
#     return tag

def get_ibe_tag(ele_num):
    """Get inside, beginning, ending tag for a paragraph or chapter.

    Args:
        num (int): The index of the element in the sequence
        seq_len (int): The total length of the sequence

    Returns:
        int: (0) for inside, (1) for beginning / boundary, (2) for ending
    """    
    if ele_num == 0:
        tag = 1
    else:
        tag = 0
    return tag

def average_embeddings(sent_embeddings):
    return [sum(parameter) / len(sent_embeddings) for parameter in zip(*sent_embeddings)] if sent_embeddings else None

def validate_paragraph(paragraph, min_par_len):
    if len(paragraph) < min_par_len or not (validate_sentence(paragraph[0]) and validate_sentence(paragraph[-1])):
        # print(f"********** Filtered paragraph is: {paragraph} ************")
        return False
    return True

def validate_sentence(sentence):
    cleaned_sent = sentence.strip()
    if cleaned_sent[0] in QUOTE_PUNCTS or cleaned_sent[-1] in QUOTE_PUNCTS:
        return False
    return True

def flatten_to_sequence(chapters, granularity):
    """Flatten embedded data structure into sequences. (Does not perform filtering.)

    Args:
        chapters (list): where each element is a dictionary representing one chapter in a text
        granularity (int): where 0 represents a sentence and 1 a paragraph

    Returns:
        list: where each element is a list of tuples representing the sentences (or entirety) of a paragraph:
            - ptag: 0 for in the middle of a paragraph, 1 for at the beginning, 2 for at the end
            - chtag: 0 for in the middle of a chapter, 1 for at the beginning, 2 for at the end
            - text: a list of sentences representing the paragraph
            - embed: a list of embeddings for each sent (or an average of the embeddings in a paragraph)
    """    
    sequence_list = []
    for ch in chapters:
        paragraphs = ch["structure"]
        num_paras = len(paragraphs)
        
        # Loop over paragraphs
        for pnum, paragraph in enumerate(paragraphs): #(par_embeds, par_text) in enumerate(zip(paragraphed_embeds, paragraphed_text)):
            ctag = get_ibe_tag(pnum)

            # If paragraph-level sequences:
            if granularity == "paragraph":
                # Append chapter tags
                pass
                #sequence_list.append((None, ctag, par_text, average_embeddings(par_embeds)))
            else:
                # Loop over sentences and tag for beginning and endings of paragraphs
                #par_len = len(par_embeds)
                num_sents = len(paragraph)
                for snum, sent in enumerate(paragraph): #(sent_embed, sent) in enumerate(zip(par_embeds, par_text)):
                    sent_embed = sent["embedding"]
                    sent_text = sent["text"]
                    
                    # Get sentence-level tag for paragraphs
                    ptag = get_ibe_tag(snum)
                    chtag = ptag if ctag == ptag else 0
                    
                    sequence_list.append((ptag, chtag, sent_text, sent_embed))
        
    return sequence_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Output sequenced file")
    parser.add_argument("--readable", dest="readable", help = "readable output file")
    parser.add_argument("--granularity", dest = "granularity", choices=["sentence", "paragraph"], default="sentence")
    parser.add_argument("--cluster", type=int, nargs = "?", help = "Cluster number to filter by")
    parser.add_argument("--filter_by_stat", dest = "filters", type=float, nargs = "*", help = "Filter by stats")
    parser.add_argument("--min_par_len", dest = "min_par_len", type = int, default = 2, help = "Min number of sentences in a paragraph")
    parser.add_argument("--seed", dest="seed", type=int)
    args, rest = parser.parse_known_args()

    if args.seed != None:
        random.seed(args.seed)
    
    print(f"Generating Sequence Now")
    with open_file(args.input, "r") as input, jsonlines.Reader(input) as reader:
        with gzip.open(args.output, mode="wt") as output, jsonlines.Writer(output) as compressed_writer:
            #with open(args.readable, mode="wt") as readable, jsonlines.Writer(readable) as debug_writer:
            for idx, doc in enumerate(tqdm(reader, desc="Generating Sequences")):
                if not args.cluster or args.cluster == int(doc["cluster"]):
                    sequence_list = flatten_to_sequence(doc["chapters"], args.granularity)
                    plabels, clabels, flattened_sentences, flattened_embeddings = zip(*sequence_list)
                    if sequence_list:
                        sequenced_text = {
                            "metadata": doc["metadata"],
                            "granularity": args.granularity,
                            "chapters": doc["chapters"],
                            "paragraph_labels": plabels,
                            "chapter_labels": clabels,
                            "flattened_sentences": flattened_sentences,
                            "flattened_embeddings": flattened_embeddings,
                            "hierarchical_labels": [[p, c] for p, c in zip(plabels, clabels)],
                            "blank": []
                        }

                        compressed_writer.write(sequenced_text)
                        