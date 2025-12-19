import argparse
from transformers import AutoModel, AutoTokenizer
import jsonlines
import torch
import gzip
from tqdm import tqdm
import logging

import json
logger = logging.getLogger(__name__)

def batch_items(items, batch_size):
    while len(items) > 0:
        yield items[:batch_size]
        items = items[batch_size:]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--output_embedding", dest="output_embedding", help="Output embedding file")
    parser.add_argument("--model_id", dest="model_id", help="Huggingface ID of model to use")
    parser.add_argument("--max_toks", type=int, dest="max_toks")
    parser.add_argument("--batch_size", type=int, default=1024, dest="batch_size")
    args, rest = parser.parse_known_args()

    logger.info(
        f"Starting text encoding process with:\n"
        f"  Input: {args.input}\n"
        f"  Output: {args.output_embedding}\n"
        f"  Model: {args.model_id}\n"
        f"  Max tokens: {args.max_toks}\n"
        f"  Batch size: {args.batch_size}"
    )

    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        logger.info("Using CPU device")

    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    model.to(device)

    with gzip.open(args.input, "rt") as input_file, gzip.open(args.output_embedding, "wt") as output_file:
        with jsonlines.Reader(input_file) as reader, jsonlines.Writer(output_file) as writer:
            for idx, doc in tqdm(enumerate(reader)):
                sents = doc["content"]
                sent_embs = []
                for sentences in batch_items(sents, args.batch_size):
                    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=args.max_toks)
                    bert_output = model(
                        input_ids = tokens["input_ids"].to(device),
                        attention_mask = tokens["attention_mask"].to(device),
                        token_type_ids = tokens["token_type_ids"].to(device),
                        output_hidden_states = True
                    )
                    bert_hidden_states = bert_output["hidden_states"]
                    sent_embs += bert_hidden_states[-1][:,0,:].tolist()

                output_embedding = {
                    "embeddings": sent_embs,
                    "source": doc["metadata"]["source"],
                    "key": doc["key"]
                }
                writer.write(output_embedding)