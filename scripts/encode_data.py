import argparse
import json
import re
from transformers import BertModel, BertTokenizer
import torch
import jsonlines
import torch
from collections import OrderedDict
from utility import make_dirs
import gzip
import json
import nltk

# Input: chapter_dict representing one chapter: key=paragraph_num, value= string paragraph_content
# Output: dict representing one chapter: key=paragraph_num, value=list of sentences in paragraph
def get_paragraphs(chapter):
    paragraphs = []
    for paragraph in chapter:
        paragraphs.append(get_sentences(paragraph)) # list of sentences
    return paragraphs

# Split paragraph text into list of sentences
def get_sentences(paragraph):
    # return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', paragraph)
    return nltk.sent_tokenize(paragraph)

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

def encode_chapter(tokenizer, model, ch_name, ch_content, max_toks):
    chapter_embeds = []
    chapter_paragraphs = get_paragraphs(ch_content)
    print(f"Chapter name: {ch_name}")
    
    # Iterate through paragraphs in chapters, grabbing sentence embeddings per paragraph
    for paragraph in chapter_paragraphs:

        result_batches = resize_batch(sentences=paragraph, batch_size=5)
        par_embeds = []
        for batch in result_batches:
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_toks)
            bert_output = model(input_ids = tokens["input_ids"].to(device),
                                attention_mask = tokens["attention_mask"].to(device),
                                token_type_ids = tokens["token_type_ids"].to(device),
                                output_hidden_states = True)

            bert_hidden_states = bert_output["hidden_states"]
            cls_token_batch = bert_hidden_states[-1][:,0,:] # dimension should be curr_batch_size, hidden_size
        # assert(len(sents) == cls_token_batch.size(0))
            s_embeddings = torch.split(cls_token_batch, split_size_or_sections=1, dim=0)
            par_embeds.extend([s_embedding.squeeze(dim=0).tolist() for s_embedding in s_embeddings])
        chapter_embeds.append(par_embeds)
    
    return (chapter_embeds, chapter_paragraphs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--model_name", dest="model_name", help="Encoder model to use")
    parser.add_argument("--max_toks", type=int, dest="max_toks")
    args, rest = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")

    torch.cuda.empty_cache()

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)
    model.to(device)

    with open(args.input, "r") as input_file, gzip.open(args.output, "w") as output_file:
        # For line in jsonlines
        input = jsonlines.Reader(input_file)
        output = jsonlines.Writer(output_file)
        enc_file_num = 0
        for idx, text in enumerate(input):
            encoded_data = {}
            encoded_data["id"] = text["id"]
            encoded_data["original_text"] = text["segments"]
            chapters = OrderedDict()
            valid_chapters = True
            for ch_name, ch_content in text["segments"].items():
                if not ch_content:
                    print("NO CONTENT FOUND")
                    valid_chapters = False
                    break
                chapters[ch_name] = encode_chapter(tokenizer, model, ch_name, ch_content, args.max_toks)
            encoded_data["encoded_segments"] = chapters
            if valid_chapters:
                output.write(encoded_data)