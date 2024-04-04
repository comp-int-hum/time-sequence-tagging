import jsonlines
from collections import OrderedDict
import argparse
import random
from utility import make_dirs
import gzip

# # Input: encoded segments grouped by paragraphs
# # Output: list of encoded segments grouped by chapter
def get_embeddings_for_chapter(encoded_chapter):
    embeddings = []
    for i in range(len(encoded_chapter)):
        p_name = "p" + str(i)
        embeddings.extend(encoded_chapter[p_name])
    return embeddings

def can_split_last(first_half, second_half):
    if len(first_half) <= 1 or len(second_half) <= 1:
        return False
    return True

# def split_chapter(encoded_chapter, fl="all"):
#     sent_embeddings = get_embeddings_for_chapter(encoded_chapter)
#     first_half = [] # list of sentence embeddings (also a list)
#     second_half = []
#     half = len(sent_embeddings) / 2
#     for i, sent_embedding in enumerate(sent_embeddings):
#         if i < half:
#             first_half.append(sent_embedding)
#         else:
#             second_half.append(sent_embedding)
    
#     if first_half and second_half:
#         if fl == "all":
#             return (first_half, second_half)
#         elif fl == "no_fl" and can_split_last(first_half, second_half):
#             return (first_half[:-1], second_half[1:])
#         elif fl == "fl" and can_split_last(first_half, second_half):
#             return (first_half[-1:], second_half[:1])
    
#     return None

def average_embeddings(sent_embeddings):
    return [sum(parameter) / len(sent_embeddings) for parameter in zip(*sent_embeddings)] if sent_embeddings else None

# def create_multiclass_datapoint(metadata, context_chapter, target_chapters, context_size):
#     # Get embedding for context chapter
#     context_embedding = average_embeddings(split_chapter(context_chapter[1], context_size)[1])
#     if not context_embedding:
#         return None
    
#     # Get embedding for target chapters and create combined embeddings for (context + possible target)
#     combined_embeddings = []
#     target_names = []
#     for ch_name, ch in target_chapters:
#         target_embedding = average_embeddings(split_chapter(ch, context_size)[0])
#         if not target_embedding:
#             return None
#         combined_embeddings.append(context_embedding + target_embedding)
#         target_names.append(ch_name)

#     # Add label and shuffle embeddings
#     labels = [0] * (len(combined_embeddings)-1) + [1]
#     assert len(labels) == len(combined_embeddings)
#     labeled_data = list(zip(combined_embeddings, labels))
#     random.shuffle(labeled_data)
#     embeddings, labels = zip(*labeled_data)

#     # Create datapoint
#     datapoint = metadata.copy()
#     datapoint["embeddings"] = embeddings
#     datapoint["labels"] = labels
#     datapoint["context_name"] = context_chapter[0]
#     datapoint["target_names"] = target_names
#     return datapoint
    
def copy_metadata(text, ch_name=[]):
    encoded_data = {}
    encoded_data["id"] = text["id"]
    encoded_data["chapter_names"] = ch_name
    return encoded_data

def get_sample_list(text_len, samples, choice_size, seed):
    random.seed(seed)
    num_samples = min(text_len-choice_size, samples)
    context_samples = random.sample(range(0, text_len-choice_size), num_samples)
    choice_samples = [random.sample(range(context+2, text_len), choice_size-1) for context in context_samples]
    return context_samples, choice_samples

def create_positive_sample_emb(chapters, context_size):
    effective_size = context_size // 2
    first_ch, second_ch = chapters
    first_ch_embeds = get_embeddings_for_chapter(first_ch)
    second_ch_embeds = get_embeddings_for_chapter(second_ch)
    if len(first_ch_embeds) < effective_size or len(second_ch_embeds) < effective_size:
        print("Context size too big for positive datapoint")
        return None
    embeds = (average_embeddings(first_ch_embeds[-effective_size:]) + average_embeddings(second_ch_embeds[:effective_size]))
    first_ch_idxs = (len(first_ch_embeds) - effective_size, len(first_ch_embeds)-1)
    second_ch_idxs = (0, effective_size - 1)
    return first_ch_idxs, second_ch_idxs, embeds
    
def create_negative_sample_emb(chapter, context_size):
    ch_embeds = get_embeddings_for_chapter(chapter)
    if len(ch_embeds) < context_size:
        print("Context size too big for negative datapoint")
        return None
    start_index = random.randint(0, len(ch_embeds)-context_size) # randomly select interval in chapter as negative datapoint
    pre_embeds = ch_embeds[start_index: start_index + (context_size // 2)]
    post_embeds = ch_embeds[start_index + (context_size // 2) : start_index + context_size]
    ch_idxs = (start_index, start_index + context_size) # inclusive, exclusive
    return ch_idxs, (average_embeddings(pre_embeds) + average_embeddings(post_embeds))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Name for datapoints file")
    parser.add_argument("--samples", type=int, dest="samples", help="Number of samples to take")
    parser.add_argument("--same", dest="same", help="True if chapter examples are from same text")
    parser.add_argument("--context_size", dest="context_size", type = int, help="Exclude first last (no_fl), only first last (fl), all") # fl stands for first last
    # parser.add_argument("--negative_size", dest="target", type = int, help="Number of negative examples")
    parser.add_argument("--seed", dest="seed", type=int)
    args, rest = parser.parse_known_args()

    random.seed(args.seed)

    print(f"Same: {args.same}")

    make_dirs(args.output)
    
    print("**Creating datapoints**")
    
    with gzip.open(args.input, "r") as input_file, gzip.open(args.output, mode="wt") as output_file:
        input = jsonlines.Reader(input_file)
        writer = jsonlines.Writer(output_file)
        # For line in jsonlines
        data = [] # datapoints for current book
        prev_book_chapters = [] # tuple list of chapters_names and chapter_contents from prev book
        for idx, text in enumerate(input):
            
            curr_book_chapters = list(text["encoded_segments"].items()) # Tuple list
            print(type(curr_book_chapters[0]))
            
            num_samples = min(len(curr_book_chapters)-1, args.samples)
            if not num_samples:
                continue
            positive_sample_indices = random.sample(range(len(curr_book_chapters)-1), num_samples) 
            positive_samples = [curr_book_chapters[i:i+2] for i in positive_sample_indices] # list of list of tuples

            if args.same == "True":
                negative_samples = random.sample(curr_book_chapters, num_samples) # remember, this is a list of tuples
            else:
                num_samples = min(len(prev_book_chapters), args.samples)
                if not num_samples:
                    continue
                negative_samples = random.sample(prev_book_chapters, num_samples)

            # print(f"positive examples: {type(positive_samples[0])}")
            for ch_lst in positive_samples:
                # print(f"ch list type: {type(ch_lst)}")
                ch_names, chs = zip(*ch_lst) # use zip* because we grab two chapters for positive
                # print(type(ch_names))
                pos_dp = copy_metadata(text, ch_names)
                pos_embeds = create_positive_sample_emb(chs, args.context_size)
                if not pos_embeds:
                    continue
                first_ch_idxs, second_ch_idxs, embeds = pos_embeds
                pos_dp["segment"] = embeds
                pos_dp["first_ch_idxs"] = first_ch_idxs
                pos_dp["second_ch_idxs"] = second_ch_idxs
                pos_dp["ground_truth"] = 1
                if embeds:
                    data.append(pos_dp)

            # print(f"Negative samples: {negative_samples}")
            # print(f"Negative: {type(negative_samples)}")
            # print(f"Negative sample: {type(negative_samples[0])}")
            for (ch_name, ch) in negative_samples:
                neg_dp = copy_metadata(text, [ch_name])
                neg_embeds = create_negative_sample_emb(ch, args.context_size)
                if not neg_embeds:
                    continue
                first_ch_idxs, embeds = neg_embeds
                neg_dp["segment"] = embeds
                neg_dp["first_ch_idxs"] = first_ch_idxs
                neg_dp["second_ch_idxs"] = None
                neg_dp["ground_truth"] = 0
                if embeds:
                    data.append(neg_dp)

            prev_book_chapters = curr_book_chapters

        random.shuffle(data)

        # Write resulting data
        for d in data:
            assert("segment" in d)
            writer.write(d)
