import argparse
import json
from transformers import BertModel, BertTokenizer
import torch
import jsonlines
import torch
from collections import OrderedDict
from utility import make_dirs
import gzip
import json
import nltk
from tqdm import tqdm

# Input: chapter_dict representing one chapter: key=paragraph_num, value= string paragraph_content
# Output: dict representing one chapter: key=paragraph_num, value=list of sentences in paragraph

def get_sentence_batch(paragraph, batch_size):
    """Given a paragraph, return the sentences batched.

    Args:
        paragraph (list): list of sentences composing one paragraph
        batch_size (int): batch size

    Returns:
        list: each element in the list is a list of sentences with the list's max len = batch_size
    """    
    current_batch = []
    batched_sentences = []

    for sent in paragraph:
        if len(current_batch) < batch_size:
            current_batch.append(sent)
        else:
            batched_sentences.append(current_batch)
            current_batch = [sent]
    
    if current_batch:
        batched_sentences.append(current_batch)

    return batched_sentences

def encode_chapter(tokenizer, model, paragraphs, max_toks):
    """Encode the sentences in each chapter provided a tokenizer, model, and sentences in each paragraph

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer for the embedding model
        model (PreTrainedModel): the embedding model
        paragraphs (list): where each element is a list of sentences representing one paragraph
        max_toks (int): the maximum length for the tokenizer

    Returns:
        list: where each element is a list of embeddings representing the sentences in a paragraph
    """    
    chapter_embeds = []
    
    # Iterate through paragraphs in chapters, grabbing sentence embeddings per paragraph
    for paragraph in paragraphs:

        batched_sentences = get_sentence_batch(paragraph=paragraph, batch_size=5)
        par_embeds = []
        
        for batch in batched_sentences:
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
    
    return chapter_embeds

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
                
    with gzip.open(args.input, "r") as input_file, gzip.open(args.output, "w") as output_file:
        with jsonlines.Reader(input_file) as reader, jsonlines.Writer(output_file) as writer:
            for idx, doc in tqdm(enumerate(reader)):
                valid_doc = True
                for chapter in doc["chapters"]:
                    if chapter["structure"]:
                        chapter["embedded_structure"] = encode_chapter(tokenizer, model, chapter["structure"], args.max_toks)
                    else:
                        # Set valid_doc to false due to encountering text structure issue
                        valid_doc = False
                        break
                
                # Check that no structural issues were encountered
                if valid_doc:
                    writer.write(doc)