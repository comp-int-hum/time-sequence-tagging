import argparse
import jsonlines 
import random
import gzip
from create_datapoints import average_embeddings
from utility import open_file

# Returns (0) for outside
#         (1) for inside
#         (2) for beginning (and for one unit sequences)
def get_iob_tag(num, seq_len):
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

def get_tag(iob_tag, beginning = True):
   return 1 if iob_tag and not beginning else iob_tag

# Sequence list: List of Tuple: (tag, embed)
# Original list: List of paragraphs/sentences
def convert_to_sequence(chapters, chapter, paragraph, granularity, beginning = True):
    sequence_list = [] # embed sequence
    original_list = [] # original text
    for (_, ch) in chapters:
        chapter_embeds, original_text = ch
        ch_len = len(chapter_embeds)

        # Loop over paragraphs
        for pnum, (par_embeds, pars) in enumerate(zip(chapter_embeds, original_text)):
            ptag = get_iob_tag(pnum, ch_len)
            par_embeds = chapter_embeds[pnum]

            # If paragraph-based tags
            if granularity:
                # Append paragraph tags
                sequence_list.append((get_tag(ptag, beginning), average_embeddings(par_embeds)))
                original_list.append(pars)
            else:
                # Loop over sentences and append tags for sentences
                par_len = len(par_embeds)
                for snum, (sent_embed, sent) in enumerate(zip(par_embeds, pars)):
                    # Check that whether sentence is first or last and determine tag based on whether ptag also first or last
                    stag = get_iob_tag(snum, par_len)
                    tag = get_tag(stag, beginning)

                    if chapter:
                        marked_sentences = ptag and (stag == ptag or par_len == 1)
                        if marked_sentences and paragraph:
                            # Set chapter boundaries labels to be different from paragraph boundary labels
                            tag = tag + 1 + int(beginning)
                        elif paragraph:
                            tag = tag
                        elif marked_sentences:
                            tag = tag
                        else:
                            tag = 0


                    # Append id to seq_list and original sent to original_list
                    sequence_list.append((tag, sent_embed))
                    original_list.append(sent)
        
    return original_list, sequence_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Output sequenced file")
    parser.add_argument("--paragraph", dest ="paragraph", type = int, help = "Mark paragraph boundaries")
    parser.add_argument("--chapter", dest = "chapter", type = int, help = "Mark chapter boundaries")
    parser.add_argument("--granularity", dest = "granularity", type = int, help = "(0) for sentence, (1) for paragraph")
    parser.add_argument("--beginning", dest ="beginning", default = 1, type = int, help="Beginning and end (1) or just inside outside (0)")
    parser.add_argument("--seed", dest="seed", type=int)
    args, rest = parser.parse_known_args()

    random.seed(args.seed)
    
    print(f"Generating Sequence Now")
    if args.paragraph and args.granularity:
        raise ValueError("Granularity cannot be paragraph-level for paragraph tagging")
    
    input = jsonlines.Reader(open_file(args.input, "r"))
    writer = jsonlines.Writer(gzip.open(args.output, mode="wt"))
    for idx, text in enumerate(input):
            sequenced_text = {"id": text["id"]}
            original_list, sequence_list = convert_to_sequence(list(text["encoded_segments"].items()), args.chapter, args.paragraph, args.granularity, args.beginning)
            sequenced_text["paragraph"] = args.paragraph
            sequenced_text["chapter"] = args.chapter
            sequenced_text["granularity"] = args.granularity
            sequenced_text["sequence"] = sequence_list
            sequenced_text["original_text"] = original_list
            writer.write(sequenced_text)





# Whether the current element is first (1), last (2), or middle (0) if seq_matters. If not, just (1) or (0).
# def get_tag(num, seq_len, boundary_context = False):
#     if boundary_context:
#         tag = 1 if not num else 2 if num == (seq_len - 1) else 0
#         tag = tag if (num != (seq_len - 1)) else 3
#     else:
#         tag = 1 if (num == 0 or num == (seq_len-1)) else 0
#         tag = tag if (num != (seq_len - 1)) else 2
#     return tag