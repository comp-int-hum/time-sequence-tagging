import argparse
import json
import re
from transformers import BertModel, BertTokenizer
import torch
import h5py
import jsonlines

def get_chapter_sentences(segment):
    all_sent = []
    paragraph_sentences = get_paragraph_sentences(segment)
    for sentences in paragraph_sentences:
        all_sent.extend(sentences)
    return all_sent

def get_paragraph_sentences(segment):
    sentences = {}
    for i, paragraph in enumerate(segment):
        sentences[i] = get_sentences(paragraph)
    return sentences

def get_sentences(paragraph):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', paragraph)

def get_batches(segment):
    chapter_s = get_chapter_sentences(segment)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--model_name", dest="model_name", help="Encoder model to use")
    parser.add_argument("--output", dest="output", help="Output files")
    parser.add_argument("--max_toks", dest="max_toks")
    args, rest = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"

    with open(args.input) as fp:
        data = json.load(fp)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)


    with jsonlines.open(args.input) as input, h5py.File(args.output, 'w') as output:
        for idx, text in enumerate(input):
            group = output.create_group(str(idx))
            for cnum, chapter in enumerate(text["segments"]):
                chapter_s = get_chapter_sentences(chapter)
                batch = tokenizer(chapter_s, padding=True, truncation=True, return_tensors="pt", max_length=args.max_toks)
                bert_output = model(input_ids = batch["input_ids"].to(device),
                                    attention_mask = batch["attention_mask"].to(device),
                                    token_type_ids = batch["token_type_ids"].to(device),
                                    output_hidden_states = True)
                
                bert_hidden_states = bert_output["hidden_states"]
                cls_token_batch = bert_hidden_states[-1][:,0,:] # dimension should be batch_size, hidden_size
                group.create_dataset(str(cnum), data=cls_token_batch.numpy())



                



    
