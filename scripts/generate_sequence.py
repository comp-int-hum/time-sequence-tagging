import argparse
import jsonlines 
import random
import gzip
from create_datapoints import average_embeddings
from utility import open_file
from tqdm import tqdm

MIN_PAR_LENGTH = 2

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


# Returns (0) for inside
#         (1) for beginning / boundary
#         (2) for ending (and for one unit sequences)
def get_ibe_tag(num, seq_len):
    tag = 0
    # if (seq_len == 1):
    #     tag = 3
    if num == 0:
        tag = 1
    elif num == (seq_len - 1):
        tag = 2
    else:
        tag = 0

    return tag

def validate_paragraph(paragraph):
    if len(paragraph) < MIN_PAR_LENGTH or not (validate_sentence(paragraph[0]) and validate_sentence(paragraph[-1])):
        # print(f"********** Filtered paragraph is: {paragraph} ************")
        return False
    return True

def validate_sentence(sentence):
    cleaned_sent = sentence.strip()
    if cleaned_sent[0] in QUOTE_PUNCTS or cleaned_sent[-1] in QUOTE_PUNCTS:
        return False
    return True

# Sequence list: List of Tuple: (tag, embed)
# Original list: List of paragraphs/sentences
def convert_to_sequence(chapters, granularity):
    sequence_list = []
    for (_, ch) in chapters:
        chapter_embeds, original_text = ch
        ch_len = len(chapter_embeds)


        # Loop over paragraphs
        for pnum, (par_embeds, pars) in enumerate(zip(chapter_embeds, original_text)):
            ctag = get_ibe_tag(pnum, ch_len)
            par_embeds = chapter_embeds[pnum]

            # If paragraph-level sequences:
            if granularity:
                # Append chapter tags
                sequence_list.append((None, ctag, pars, average_embeddings(par_embeds)))
            else:
                # Loop over sentences and tag for beginning and endings of paragraphs
                par_len = len(par_embeds)
                for snum, (sent_embed, sent) in enumerate(zip(par_embeds, pars)):
                    # Get sentence-level tag for paragraphs
                    ptag = get_ibe_tag(snum, par_len)
                    chtag = ptag if ctag == ptag else 0

                    # Append id to seq_list and original sent to original_list
                    sequence_list.append((ptag, chtag, sent, sent_embed))
        
    return sequence_list


def filter_by_paragraph_and_convert_to_sequence(chapters, granularity):
    sequence_list = []
    for (_, ch) in chapters:
        paragraphs_list = []
        chapter_embeds, original_text = ch
        ch_len = len(chapter_embeds)

        # Loop over paragraphs
        for pnum, (par_embeds, par) in enumerate(zip(chapter_embeds, original_text)):

            ctag = get_ibe_tag(pnum, ch_len)
            par_embeds = chapter_embeds[pnum]

            if validate_paragraph(par):
                # If paragraph-level sequences:
                if granularity:
                    # Append chapter tags
                    paragraphs_list.append((None, ctag, par, average_embeddings(par_embeds)))
                else:
                    # Loop over sentences and tag for beginning and endings of paragraphs
                    par_len = len(par_embeds)
                    for snum, (sent_embed, sent) in enumerate(zip(par_embeds, par)):
                        # Get sentence-level tag for paragraphs
                        ptag = get_ibe_tag(snum, par_len)
                        chtag = ptag if ctag == ptag else 0

                        # Append id to seq_list and original sent to original_list
                        paragraphs_list.append((ptag, chtag, sent, sent_embed))
            elif pnum in [0, ch_len-1]:
                paragraphs_list = []
                break
        
        sequence_list.extend(paragraphs_list)

    return sequence_list

def filter_by_sentence_and_convert_to_sequence(chapters, granularity):
    sequence_list = []
    for (_, ch) in chapters:
        paragraphs_list = []
        chapter_embeds, original_text = ch
        ch_len = len(chapter_embeds)

        if ch_len < 3:
            continue
        # Loop over paragraphs
        for pnum, (par_embeds, par) in enumerate(zip(chapter_embeds, original_text)):

            ctag = get_ibe_tag(pnum, ch_len)
            par_embeds = chapter_embeds[pnum]

            subparagraph_list = []
            valid_par = validate_paragraph(par)

            if valid_par:
                # If paragraph-level sequences:
                if granularity:
                    # Append chapter tags
                    subparagraph_list.append((None, ctag, par, average_embeddings(par_embeds)))
                else:
                    # Loop over sentences and tag for beginning and endings of paragraphs
                    par_len = len(par_embeds)
                    for snum, (sent_embed, sent) in enumerate(zip(par_embeds, par)):
                        # Get sentence-level tag for paragraphs
                        ptag = get_ibe_tag(snum, par_len)
                        chtag = ptag if ctag == ptag else 0

                        # Append id to seq_list and original sent to original_list
                        if validate_sentence(sent):
                            subparagraph_list.append((ptag, chtag, sent, sent_embed))

            if valid_par and (granularity != (len(subparagraph_list) >= MIN_PAR_LENGTH)):
                paragraphs_list.extend(subparagraph_list)
            elif pnum in [0, ch_len-1]:
                paragraphs_list = []
                break
        
        sequence_list.extend(paragraphs_list)

    # if sequence_list:
    #     ptags, chtags, _, _ = zip(*sequence_list)
    #     assert ptags.count(1) == ptags.count(2)
    #     print(f"Start chapter tags: {chtags.count(1)} -- End chapter tags: {chtags.count(2)}")
    #     assert chtags.count(1) == chtags.count(2)
    return sequence_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Output sequenced file")
    parser.add_argument("--granularity", dest = "granularity", type = int, help = "(0) for sentence, (1) for paragraph")
    parser.add_argument("--seed", dest="seed", type=int)
    args, rest = parser.parse_known_args()

    random.seed(args.seed)
    
    print(f"Generating Sequence Now")
    
    input = jsonlines.Reader(open_file(args.input, "r"))
    writer = jsonlines.Writer(gzip.open(args.output, mode="wt"))
    for idx, doc in enumerate(tqdm(input, desc="Generating Sequences")):  
        print(f"Doc keys: {doc.keys()}")
    
    input.close()
    writer.close()
        # sequence_list = filter_by_paragraph_and_convert_to_sequence(list(doc["encoded_segments"].items()), args.granularity)
        # if sequence_list:
        #     paragraph_labels, chapter_labels, text, embeddings = zip(*sequence_list)
        #     sequenced_text = {"id": doc["id"],
        #                     "granularity" : args.granularity,
        #                     "paragraph_labels": paragraph_labels,
        #                     "chapter_labels": chapter_labels,
        #                     "text": text,
        #                     "embeddings": embeddings}
        #     writer.write(sequenced_text)