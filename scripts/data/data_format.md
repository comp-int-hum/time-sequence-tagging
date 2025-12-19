# Base structured data format
```json
{
  "metadata": {
    "title": "<str>",
    "author": "<str>",
    "text_id": "<str>",
    "segment_num": <int> (optional),
    "year": None,
    "source": "<str>"
  },
  "structure": {
    "type": "chapter",
    "label": "chapter_name",
    "subunits": [
        {
            "type": "paragraph",
            "label": None,
            "subunits": [
                {
                    "type": "sentence",
                    "label": None,
                    "text": sentence
                }
            ]
        }
    ]
  },
}
```

# Hierarchical Text Data Format

Each data entry represents a structured passage with hierarchical segmentation labels

# Structure

```json
{
  "metadata": {
    "title": "<str>",
    "author": "<str>",
    "text_id": "<str>",
    "segment_num": <int> (optional),
    "year": None,
    "source": "<str>"
  },
  "content": ["<unit_1>", "<unit_2>", "..."],
  "base_unit": "<str>",  // for example: "sentences", "paragraphs"
  "labels": {
    "<level_1_name>": [0, 0, 1, ...],
    "<level_2_name>": [1, 0, 0, ...]
  },
  "hierarchy_order": ["<base_unit>", "<level_1>", "<level_2>", ...],
  "sequence_length": 100,
  "embedded_units": []
}
```

Train a word2vec on whatever data you have; consider removing embeddings. If you don't have an embedding model, train one naively.


# New split data structure

embedding:
```json
   {
      "embeddings": sent_embs,
      "source": doc["metadata"]["source"],
      "key": doc["key"]
   }
```

data_content:
```json
    {
        "metadata": doc["metadata"],
        "content": flat_text,
        "key": doc_num,
        # Extra
        "structure": structure,
    }
```

labels:
```json
    {
        "base_unit": hierarchy_order[0],
        "labels": {
            "chapters": [1, 0, 0, 0, ...],
            "paragraphs": [1, 0, 1, 0, ...],
        },
        "hierarchy_order": ["sentences", "paragraphs", "chapters"],
        "source": doc["metadata"]["source"],
        "key": doc_num
    }
```

