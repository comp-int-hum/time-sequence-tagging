import jsonlines
from collections import OrderedDict
import argparse
import random
from utility import make_dirs

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

def split_chapter(encoded_chapter, fl="all"):
    sent_embeddings = get_embeddings_for_chapter(encoded_chapter)
    first_half = [] # list of sentence embeddings (also a list)
    second_half = []
    half = len(sent_embeddings) / 2
    for i, sent_embedding in enumerate(sent_embeddings):
        if i < half:
            first_half.append(sent_embedding)
        else:
            second_half.append(sent_embedding)
    
    if first_half and second_half:
        if fl == "all":
            return (first_half, second_half)
        elif fl == "no_fl" and can_split_last(first_half, second_half):
            return (first_half[:-1], second_half[1:])
        elif fl == "fl" and can_split_last(first_half, second_half):
            return (first_half[-1:], second_half[:1])
    
    return None

def get_first_half(encoded_chapter):
    return split_chapter(encoded_chapter)[0]

def get_second_half(encoded_chapter):
    return split_chapter(encoded_chapter)[1]

def average_embeddings(sent_embeddings):
    return [sum(parameter) / len(sent_embeddings) for parameter in zip(*sent_embeddings)] if sent_embeddings else None

# input: chapter num and encoded_text

# input: metadata is a dict
def create_binary_datapoint_pair(metadata, prev_ch, next_ch, prev_ch_n, next_ch_n, fl):
    prev = split_chapter(prev_ch, fl)
    next = split_chapter(next_ch, fl)
    if not prev or not next:
        return None
    
    positive_dp = create_binary_datapoint(metadata, prev[1], next[0], prev_ch_n, next_ch_n, True)
    negative_dp = create_binary_datapoint(metadata, next[0], next[1], next_ch_n, next_ch_n, False)
    
    if positive_dp and negative_dp:
        return [positive_dp, negative_dp]
    return None

def create_multiclass_datapoint(metadata, context_chapter, target_chapters, context_size):
    # Get embedding for context chapter
    context_embedding = average_embeddings(split_chapter(context_chapter[1], context_size)[1])
    if not context_embedding:
        return None
    
    # Get embedding for target chapters and create combined embeddings for (context + possible target)
    combined_embeddings = []
    target_names = []
    for ch_name, ch in target_chapters:
        target_embedding = average_embeddings(split_chapter(ch, context_size)[0])
        if not target_embedding:
            return None
        combined_embeddings.append(context_embedding + target_embedding)
        target_names.append(ch_name)

    # Add label and shuffle embeddings
    labels = [0] * (len(combined_embeddings)-1) + [1]
    assert len(labels) == len(combined_embeddings)
    labeled_data = list(zip(combined_embeddings, labels))
    random.shuffle(labeled_data)
    embeddings, labels = zip(*labeled_data)

    # Create datapoint
    datapoint = metadata.copy()
    datapoint["embeddings"] = embeddings
    datapoint["labels"] = labels
    datapoint["context_name"] = context_chapter[0]
    datapoint["target_names"] = target_names
    return datapoint
  
def create_binary_datapoint(metadata, prev, next, prev_ch_n, next_ch_n, positive):
    emb_prev = average_embeddings(prev)
    emb_next = average_embeddings(next)

    if emb_prev and emb_next:
        dp = metadata.copy()
        emb_prev.extend(emb_next)
        dp["embeddings"] = emb_prev
        assert(dp["embeddings"])
        dp["prev_chapter_name"] = prev_ch_n
        dp["next_chapter_names"] = next_ch_n
        dp["data_type"] = positive
        return dp
    
    return None
    
# # Input: number of times to sample
# # Output: samples
# def sample_chapters(num_samples, encoded_text):
#     num_chapters = len(encoded_text)
#     random_samples = random.sample(range(0, num_chapters-1), num_samples)

def get_metadata(text):
    encoded_data = {}
    encoded_data["title"] = text["title"]
    encoded_data["author"] = text["author"]
    encoded_data["edition"] = text["edition"]
    encoded_data["pub_info"] = text["pub_info"]
    encoded_data["tags"] = text["tags"]
    return encoded_data

def get_sample_list(text_len, samples, choice_size, seed):
    random.seed(seed)
    num_samples = min(text_len-choice_size, samples)
    context_samples = random.sample(range(0, text_len-choice_size), num_samples)
    choice_samples = [random.sample(range(context+2, text_len), choice_size-1) for context in context_samples]
    return context_samples, choice_samples

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Name for datapoints file")
    parser.add_argument("--samples", type=int, dest="samples", help="Number of samples to take")
    parser.add_argument("--same_text", dest="same", help="True if chapter examples are from same text")
    parser.add_argument("--context_size", dest="context_size", help="Exclude first last (no_fl), only first last (fl), all") # fl stands for first last
    parser.add_argument("--target_size", dest="target", type = int, help="Number of choices for pred")
    parser.add_argument("--seed", dest="seed", type=int)
    args, rest = parser.parse_known_args()

    print(f"Same: {args.same}")

    make_dirs(args.output)
    assert(args.target_size >= 2)
    
    with jsonlines.open(args.input, "r") as input, jsonlines.open(args.output, mode="w") as writer:
        # For line in jsonlines
        data = [] # datapoints for current book
        prev_book_chapters = [] # tuple list of chapters_names and chapter_contents from prev book
        for idx, text in enumerate(input):
            metadata = get_metadata(text)
            
            curr_book_chapters = list(text["encoded_segments"].items())

            context_samples, target_samples = get_sample_list(len(curr_book_chapters), args.samples, args.target, args.seed)

            for (cnum, tnums) in zip(context_samples, target_samples):
                context_chapter = curr_book_chapters[cnum][1]
                
                if args.same == "True":
                    target_chapters = [curr_book_chapters[tnum] for tnum in tnums].extend([context_chapter])
                else:
                    if len(prev_book_chapters) >= args.target:
                        choice_samples = random.sample(range(0, len(prev_book_chapters)), args.target)
                        target_chapters = [prev_book_chapters[c][1] for c in choice_samples].extend([context_chapter])
                    else:
                        continue
                
                # dps = create_binary_datapoint_pair(metadata, 
                dps = create_multiclass_datapoint(metadata, context_chapter, target_chapters, args.context_size)
                if dps:
                    data.append(dps) # add to overall datapoints
                
            prev_book_chapters = curr_book_chapters

        random.shuffle(data)

        # Write resulting data
        for d in data:
            assert("embeddings" in d)
            writer.write(d)
