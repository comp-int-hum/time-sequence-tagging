import argparse
import json
import re
from transformers import BertModel, BertTokenizer
import torch
import h5py
import jsonlines
import torch
from collections import OrderedDict

# Input: chapter_dict representing one chapter: key=paragraph_num, value = string paragraph_content
# Output: list of all sentences in the chapter
def get_chapter_sentences(chapter_dict):
    all_sent = []
    paragraph_sentences = get_paragraph_sentences(chapter_dict) # dictionary
    for sentences in paragraph_sentences.values():
        all_sent.extend(sentences)
    return all_sent

# Input: chapter_dict representing one chapter: key=paragraph_num, value= string paragraph_content
# Output: dict representing one chapter: key=paragraph_num, value=list of sentences in paragraph
def get_paragraph_sentences(chapter_dict):
    paragraphs = OrderedDict()
    for pnum, paragraph in enumerate(list(chapter_dict.values())):
        paragraphs[pnum] = get_sentences(paragraph) # list of sentences
    # print(paragraphs)
    return paragraphs

# Split paragraph text into list of sentences
def get_sentences(paragraph):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', paragraph)

def save_dict_to_hdf5(pointer, dict):
    if dict:
        for key, val in dict.items():
            pointer.create_group(key)
            pointer[key] = val



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--model_name", dest="model_name", help="Encoder model to use")
    parser.add_argument("--output", dest="output", help="Output files")
    parser.add_argument("--max_toks", type=int, dest="max_toks")
    args, rest = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)

    with jsonlines.open(args.input, "r") as input, open(args.output, mode="w") as output:
        # duplicates = set()
        encoded_data = {}

        # For line in jsonlines
        for idx, text in enumerate(input):
            # if text["title"] in duplicates:
            #     continue
            # else:
            #     duplicates.add(text["title"])
            chapters = []
            for cnum, (ch_name, ch_content) in enumerate(text["segments"].items()):
                chapter = {}
                chapter_by_par = get_paragraph_sentences(ch_content)
                
                # Iterate through paragraphs in chapters, grabbing sentence embeddings per paragraph
                for pnum, sents in chapter_by_par.items():

                    batch = tokenizer(sents, padding=True, truncation=True, return_tensors="pt", max_length=args.max_toks)
                    bert_output = model(input_ids = batch["input_ids"].to(device),
                                        attention_mask = batch["attention_mask"].to(device),
                                        token_type_ids = batch["token_type_ids"].to(device),
                                        output_hidden_states = True)
            
                    bert_hidden_states = bert_output["hidden_states"]
                    cls_token_batch = bert_hidden_states[-1][:,0,:] # dimension should be batch_size, hidden_size
                    assert(len(sents) == cls_token_batch.size(0))
                    s_embeddings = torch.split(cls_token_batch, split_size_or_sections=1, dim=0)
                    chapter[str(pnum)] = [s_embedding.squeeze(dim=0).tolist() for s_embedding in s_embeddings]
                
                chapters.append(chapter)

                # chapter_folder = group.create_group(str(cnum))
                # chapter_folder.create_dataset(str(ch_name), data=cls_token_batch.numpy())
            encoded_data[text["title"]] = chapters
        json.dump(encoded_data, output)

    # with jsonlines.open(args.input) as input, h5py.File(args.output, 'w') as output:
    #     duplicates = set()
    #     for idx, text in enumerate(input):
    #         if text["title"] in duplicates:
    #             continue
    #         else:
    #             duplicates.add(text["title"])
    #         # Create groups
    #         group = output.create_group(text["title"])
    #         author = group.create_group("author")
    #         edition = group.create_group("edition")
    #         pub_info = group.create_group("pub_info")

    #         # Assign values to groups
    #         # group["author"] = text["author"]
    #         author.create_dataset(text["author"])
    #         # group["edition"] = text["edition"]
    #         # save_dict_to_hdf5(pub_info, text["pub_info"])

    #         for cnum, ch_name, ch_content in enumerate(text["segments"].items()):
    #             chapter_s = get_chapter_sentences(ch_content)
    #             batch = tokenizer(chapter_s, padding=True, truncation=True, return_tensors="pt", max_length=args.max_toks)
    #             bert_output = model(input_ids = batch["input_ids"].to(device),
    #                                 attention_mask = batch["attention_mask"].to(device),
    #                                 token_type_ids = batch["token_type_ids"].to(device),
    #                                 output_hidden_states = True)
            
    #             bert_hidden_states = bert_output["hidden_states"]
    #             cls_token_batch = bert_hidden_states[-1][:,0,:] # dimension should be batch_size, hidden_size
    #             chapter_folder = group.create_group(str(cnum))
    #             chapter_folder.create_dataset(str(ch_name), data=cls_token_batch.numpy())



                



    
