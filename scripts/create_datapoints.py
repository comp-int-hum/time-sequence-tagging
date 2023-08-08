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

def split_chapter(encoded_chapter):
    sent_embeddings = get_embeddings_for_chapter(encoded_chapter)
    first_half = [] # list of sentence embeddings (also a list)
    second_half = []
    half = len(sent_embeddings) / 2
    for i, sent_embedding in enumerate(sent_embeddings):
        if i < half:
            first_half.append(sent_embedding)
        else:
            second_half.append(sent_embedding)

    return (first_half, second_half)

def get_first_half(encoded_chapter):
    return split_chapter(encoded_chapter)[0]

def get_second_half(encoded_chapter):
    return split_chapter(encoded_chapter)[1]

def average_embeddings(sent_embeddings):
    return [sum(parameter) / len(sent_embeddings) for parameter in zip(*sent_embeddings)]

# input: chapter num and encoded_text

# input: metadata is a dict
def create_datapoint(metadata, first, second):
    datapoint = metadata.copy()
    datapoint["first"] = average_embeddings(first)
    datapoint["second"] = average_embeddings(second)
    return datapoint

# # Input: number of times to sample
# # Output: samples
# def sample_chapters(num_samples, encoded_text):
#     num_chapters = len(encoded_text)
#     random_samples = random.sample(range(0, num_chapters-1), num_samples)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Encoded file")
    parser.add_argument("--output", dest="output", help="Name for datapoints file")
    parser.add_argument("--samples", type=int, dest="samples", help="Number of samples to take")
    args, rest = parser.parse_known_args()
    
    with jsonlines.open(args.input, "r") as input, jsonlines.open(args.output, mode="w") as writer:
        # For line in jsonlines
        data = []
        for idx, text in enumerate(input):
            encoded_data = {}
            encoded_data["title"] = text["title"]
            encoded_data["author"] = text["author"]
            encoded_data["edition"] = text["edition"]
            encoded_data["pub_info"] = text["pub_info"]
            
            chapters = list(text["encoded_segments"].values())
            chapter_names = list(text["encoded_segments"].values())
            num_chapters = len(chapters)
            num_samples = min(num_chapters-1, args.samples)
            sample_list = random.sample(range(0, num_chapters-1), num_samples)


            for cnum in sample_list:
                first_ch = split_chapter(chapters[cnum])
                positive_dp = create_datapoint(encoded_data, first=first_ch[1], second = get_first_half(chapters[cnum+1]))
                negative_dp = create_datapoint(encoded_data, first=first_ch[0], second=first_ch[1])
                positive_dp["first_name"] = chapter_names[cnum]
                positive_dp["second_name"] = chapter_names[cnum+1]
                negative_dp["first_name"] = chapter_names[cnum]
                negative_dp["second_name"] = chapter_names[cnum]
                positive_dp["positive"] = True
                negative_dp["positive"] = False
                data.append(positive_dp)
                data.append(negative_dp)

        random.shuffle(data)

        # Write resulting data
        for d in data:
            writer.write(d)