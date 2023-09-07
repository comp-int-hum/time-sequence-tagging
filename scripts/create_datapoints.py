import jsonlines
from collections import OrderedDict
import argparse
import random
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

def split_chapter(encoded_chapter, fl):
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
        if fl == "inc_fl":
            return (first_half, second_half)
        elif fl == "no_fl" and can_split_last(first_half, second_half):
            return (first_half[:-1], second_half[1:])
        elif fl == "only_fl" and can_split_last(first_half, second_half):
            return (first_half[-1:], second_half[:1])
    
    return None

def get_first_half(encoded_chapter):
    return split_chapter(encoded_chapter)[0]

def get_second_half(encoded_chapter):
    return split_chapter(encoded_chapter)[1]

def average_embeddings(sent_embeddings):
    return [sum(parameter) / len(sent_embeddings) for parameter in zip(*sent_embeddings)]

# input: chapter num and encoded_text

# input: metadata is a dict
def create_datapoint_pair(metadata, prev_ch, next_ch, prev_ch_n, next_ch_n, fl):
    prev = split_chapter(prev_ch, fl)
    next = split_chapter(next_ch, fl)
    if not prev or not next:
        return None
    
    positive_dp = create_datapoint(metadata, prev[1], next[0], prev_ch_n, next_ch_n, True)
    negative_dp = create_datapoint(metadata, next[0], next[1], next_ch_n, next_ch_n, False)
    
    if positive_dp and negative_dp:
        return [positive_dp, negative_dp]
    return None

def create_datapoint(metadata, prev, next, prev_ch_n, next_ch_n, positive):
    emb_prev = average_embeddings(prev)
    emb_next = average_embeddings(next)

    if emb_prev and emb_next:
        dp = metadata.copy()
        emb_prev.extend(emb_next)
        dp["embeddings"] = emb_prev
        assert(dp["embeddings"])
        dp["first_name"] = prev_ch_n
        dp["second_name"] = next_ch_n
        dp["positive"] = positive
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
    return encoded_data

def get_sample_list(text_len, samples):
    num_samples = min(text_len-1, samples)
    return random.sample(range(0, text_len-1), num_samples)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Name for datapoints file")
    parser.add_argument("--samples", type=int, dest="samples", help="Number of samples to take")
    parser.add_argument("--same", dest="same", help="True if chapter examples are from same text")
    parser.add_argument("--fl", dest="fl", help="Whether to include last sentence; choices: no_fl, only_fl, inc_fl") # fl stands for first last
    args, rest = parser.parse_known_args()

    print(f"Same: {args.same}")
    
    with jsonlines.open(args.input, "r") as input, jsonlines.open(args.output, mode="w") as writer:
        # For line in jsonlines
        data = [] # datapoints for current book
        past_chapters = [] # chapters from prev book
        past_names = []
        for idx, text in enumerate(input):
            metadata = get_metadata(text)
            
            chapters = list(text["encoded_segments"].values())
            chapter_names = list(text["encoded_segments"].values())

            sample_list = get_sample_list(len(chapters), args.samples)

            for cnum in sample_list:

                next_chapter = chapters[cnum+1]
                next_ch_n = chapter_names[cnum+1]

                if args.same == "True":
                    prev_chapter = chapters[cnum]
                    prev_ch_n = chapter_names[cnum]

                elif past_chapters:
                    rand_past = random.randint(0, len(past_chapters)-1)
                    prev_chapter = past_chapters[rand_past]
                    prev_ch_n = past_names[rand_past]
                else:
                    continue

                dps = create_datapoint_pair(metadata, prev_chapter, next_chapter, prev_ch_n, next_ch_n, args.fl)
                if dps:
                    data.extend(dps) # add to overall datapoints
                
            past_chapters = chapters
            past_names = chapter_names

        random.shuffle(data)

        # Write resulting data
        for d in data:
            assert("embeddings" in d)
            writer.write(d)
