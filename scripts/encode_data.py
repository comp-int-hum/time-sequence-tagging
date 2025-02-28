import argparse
from transformers import AutoModel, AutoTokenizer
import torch
import jsonlines
import torch
import gzip
from tqdm import tqdm


def batch_items(items, batch_size):
    while len(items) > 0:
        yield items[:batch_size]
        items = items[batch_size:]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="File containing gutenberg & women writers project")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--model_id", dest="model_id", help="Huggingface ID of model to use")
    parser.add_argument("--max_toks", type=int, dest="max_toks")
    parser.add_argument("--batch_size", type=int, default=1024, dest="batch_size")
    args, rest = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")

    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    model.to(device)
                
    with gzip.open(args.input, "r") as input_file, gzip.open(args.output, "w") as output_file:
        with jsonlines.Reader(input_file) as reader, jsonlines.Writer(output_file) as writer:
            for idx, doc in tqdm(enumerate(reader)):
                sents = sum([sum(c["structure"], []) for c in doc["chapters"]], [])
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

                counter = 0
                for i in range(len(doc["chapters"])):
                    for j in range(len(doc["chapters"][i]["structure"])):
                        for k in range(len(doc["chapters"][i]["structure"][j])):
                            doc["chapters"][i]["structure"][j][k] = {
                                "text" : doc["chapters"][i]["structure"][j][k],
                                "embedding" : sent_embs[counter]
                            }
                            counter += 1
                writer.write(doc)
