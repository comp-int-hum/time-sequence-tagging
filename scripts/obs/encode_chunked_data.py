import argparse
import json
import re
from transformers import BertModel, BertTokenizer
import torch
import h5py
import jsonlines
import torch
from collections import OrderedDict
from utility import make_dirs
import gzip
from utility import make_dirs
import json

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

def resize_batch(sentences, batch_size):
    current_batch = []
    result_batches = []

    for sent in sentences:
        if len(current_batch) < batch_size:
            current_batch.append(sent)
        else:
            result_batches.append(current_batch)
            current_batch = [sent]
    
    if current_batch:
        result_batches.append(current_batch)

    return result_batches

def encode_chapter(tokenizer, model, ch_name, ch_content):
    chapter = {}
    chapter_by_par = get_paragraph_sentences(ch_content)
    print(f"Chapter name: {ch_name}")
    
    # Iterate through paragraphs in chapters, grabbing sentence embeddings per paragraph
    for pnum, sents in chapter_by_par.items():

        result_batches = resize_batch(sentences=sents, batch_size=5)
        par_name = "p" + str(pnum)
        chapter[par_name] = []
        for batch in result_batches:
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=args.max_toks)
            bert_output = model(input_ids = tokens["input_ids"].to(device),
                                attention_mask = tokens["attention_mask"].to(device),
                                token_type_ids = tokens["token_type_ids"].to(device),
                                output_hidden_states = True)

            bert_hidden_states = bert_output["hidden_states"]
            cls_token_batch = bert_hidden_states[-1][:,0,:] # dimension should be curr_batch_size, hidden_size
        # assert(len(sents) == cls_token_batch.size(0))
            s_embeddings = torch.split(cls_token_batch, split_size_or_sections=1, dim=0)
            chapter[par_name].extend([s_embedding.squeeze(dim=0).tolist() for s_embedding in s_embeddings])
    
    return chapter

def new_file_name(base, file_num):
    f"{base}_{file_num}.jsonl.gz"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--model_name", dest="model_name", help="Encoder model to use")
    parser.add_argument("--output_names", dest="output_name", help="Output files")
    parser.add_argument("--registry", dest="registry", help="Output files")
    parser.add_argument("--max_toks", type=int, dest="max_toks")
    parser.add_argument("--max_len", type = int, dest = "max_len")
    args, rest = parser.parse_known_args()

    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"

    # print(f"Device: {device}")

    make_dirs(args.registry)

    torch.cuda.empty_cache()

    device = "cuda"

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)
    model.to(device)

    with jsonlines.open(args.input, "r") as input:
        # For line in jsonlines
        enc_file_num = 0
        output_filepaths = []
        output_file = gzip.open(output_filepaths[0], mode="w")
        for idx, text in enumerate(input):
            if idx % args.max_len == 0:
                if idx != 0:
                    output_file.close()
                    enc_file_num += 1
                output_filepaths.append(new_file_name(args.output_name, enc_file_num))
                output_file = gzip.open(output_filepaths[-1], mode="w")
            encoded_data = {}
            encoded_data["id"] = text["id"]
            chapters = OrderedDict()
            valid_chapters = True
            for ch_name, ch_content in text["segments"].items():
                if not ch_content:
                    print("NO CONTENT FOUND")
                    valid_chapters = False
                    break
                chapters[ch_name] = encode_chapter(tokenizer, model, ch_name, ch_content)
            encoded_data["encoded_segments"] = chapters
            if valid_chapters:
                output_file.write(encoded_data)
        if idx % args.max_len < int(args.max_len * 0.8):
            output_filepaths = output_filepaths[:-1]
        with open(args.registry, mode="w") as registry:
            json.dump(output_filepaths, registry)
            

        


#### _______________H5PY TEST CODE __________________
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