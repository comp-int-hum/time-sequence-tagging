import argparse
import jsonlines 
import random
import gzip
from create_datapoints import average_embeddings

# Whether the current element is first (1), last (2), or middle (0) if seq_matters. If not, just (1) or (0).
# def get_tag(num, seq_len, boundary_context = False):
#     if boundary_context:
#         tag = 1 if not num else 2 if num == (seq_len - 1) else 0
#         tag = tag if (num != (seq_len - 1)) else 3
#     else:
#         tag = 1 if (num == 0 or num == (seq_len-1)) else 0
#         tag = tag if (num != (seq_len - 1)) else 2
#     return tag

def get_tag(num, seq_len, boundary_context = False):
    tag = 0
    if (seq_len == 1):
        tag = 3
    elif num == 0:
        tag = 1
    elif num == (seq_len - 1):
        tag = 2
    else:
        tag = 0

    if not boundary_context:
        if (tag == 2 or tag == 3):
            tag = 1
    return tag

# Sequence list: List of Tuple: (tag, embed)
# Original list: List of paragraphs/sentences
def convert_to_sequence(chapters, paragraph=0, boundary_context=True):
    sequence_list = [] # embed sequence
    original_list = [] # original text
    for (_, ch) in chapters:
        chapter_embeds, original_text = ch
        ch_len = len(chapter_embeds)

        # Loop over paragraphs
        for pnum, (par_embeds, pars) in enumerate(zip(chapter_embeds, original_text)):
            ptag = get_tag(pnum, ch_len, boundary_context)
            par_embeds = chapter_embeds[pnum]

            # If paragraph-based tags
            if paragraph:
                # Append paragraph tags
                sequence_list.append((ptag, average_embeddings(par_embeds)))
                original_list.append(pars)
            else:
                # Loop over sentences and append tags for sentences
                par_len = len(par_embeds)
                for snum, (sent_embed, sent) in enumerate(zip(par_embeds, pars)):
                    # Check that whether sentence is first or last and determine tag based on whether ptag also first or last
                    stag = get_tag(snum, par_len, boundary_context)
                    stag = stag if stag == ptag else 0
                    print(f"Ptag: {ptag} - Stag: {stag}")

                    # Append id to seq_list and original sent to original_list
                    sequence_list.append((stag, sent_embed))
                    original_list.append(sent)
    print(f"Sequence list: {sequence_list}")          
    return original_list, sequence_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Output sequenced file")
    parser.add_argument("--paragraph", dest ="par", type = int, help="Paragraph (1) or sentence (0) sequence")
    parser.add_argument("--boundary_context", dest ="boundary_context", default = 1, type = int, help="Boundary context (1) or no context (0)")
    parser.add_argument("--seed", dest="seed", type=int)
    args, rest = parser.parse_known_args()

    random.seed(args.seed)
    
    print(f"Generating Sequence Now")

    with gzip.open(args.input, "r") as input_file, gzip.open(args.output, mode="wt") as output_file:
            input = jsonlines.Reader(input_file)
            writer = jsonlines.Writer(output_file)
            
            for idx, text in enumerate(input):
                 sequenced_text = {"id": text["id"]}
                 original_list, sequence_list = convert_to_sequence(list(text["encoded_segments"].items()), args.par, args.boundary_context)
                 sequenced_text["paragraph"] = args.par
                 sequenced_text["sequence"] = sequence_list
                 sequenced_text["original_text"] = original_list
                 writer.write(sequenced_text)
