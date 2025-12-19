To run:
	pip install -r requirements.txt
	scons -Q

# Pipeline structure
1. Process data sources into a standard structured format (see example below):
	1. Current scripts for each dataset:
		1. Chicago:
			1. `extract_structure_from_chicago.py`
		2. Gutenberg:
			1. `fetch_from_gutenberg.py`
			2. `extract_structure_from_gutenberg.py`
		3. Chapterbreak:
			1. `restructure_chapterbreak.py`
2. Preprocess the standard structured format (filtering sentences based on regex pattern arguments, merging paragraphs if needed, discarding short chapters, etc.). This also serves as a sanity check and will verify that the structure of the current document is correct (all types within a given level are the same — no "chapter" mixed with "sentence" at the same level)
		1. `text_structure_cleaner.py`
3. Retrieve hierarchical labels and convert each dataset's structured format file into a `data_content` file and a `labels` file (see formats below).
	1. `process_structured_text.py`
4. Encode the `data_content` file for each dataset to also get a `embeddings` file (see format below).
	1. `encode_data.py`
5. Generate train/dev/test splits:
	1. `generate_splits.py`
	2. Will result in 9 outputs:
		1. Train files: `dataset_name/data.train.gz`, `dataset_name/labels.train.gz`, `dataset_name/embeddings.train.gz`
		2. Dev files: `dataset_name/data.dev.gz`, `dataset_name/labels.dev.gz`, `dataset_name/embeddings.dev.gz
		3. Test files: `dataset_name/data.test.gz`, `dataset_name/labels.test.gz`, `dataset_name/embeddings.test.gz
6. Train the model for sequence tagging
	1. `train_hrnn.py` --> currently can train `HRNN`, `RNNTagger`, and `SimpleTagger`
7. Plot training metrics
	1. `plot_training_metrics.py`
8. Compute ROC metrics to determine best thresholds
	1. `compute_roc_metrics.py`
9. Compute confidence metrics (to see examples of true positives, false positives, false negatives, true negatives, etc.)
	1. `compute_text_confidence_metrics.py`

# Incomplete tasks
- Implement `HRNN` for language modeling
- Refactor `build_visualizations.py` and `evaluate.py`
- Delete `train_sequence_model.py`
- Should simplify RNN tagger implementations by just setting `teacher_forcing` to `1.0`
- Unclear whether `build_visualizations` (the sequence length accuracy visualizations) is currently working —> in-progress
- Unclear whether `evaluate.py` (passing in a model and eval data files and returning guesses and golds) is currently working --> in progress

# Other important files
- `batch_utils.py`: handles all the typical batching functions related to padding

# Data formats
## General structure format
```json
# General unified structure format (output of step 1)
{
	"metadata":
		{
			"title": ...,
			"author": ...,
			"text_id": ..., # <str>
			"segment_num": ..., # <int> optional
			"year": ...,
			"source": ...,
		}
	"structure":
		[
			{
				"type": "chapter",
				"label": "Chapter 1",
				"subunits": 
					[
						{
							"type": "paragraph",
							"label": None,
							"subunits":
								[
									{
										"type": "sentence",
										"label": None,
										"text": "Sentence 1 in the paragraph"
									}
								]
						},
						{
							"type": "paragraph",
							"label": None,
							"subunits":
								[
									{
										"type": "sentence",
										"label": None,
										"text": "Sentence 2 in the paragraph"
									}
								]
						}
					]
			 },
			 {
				"type": "chapter",
				"label": "Chapter 2",
				"subunits": 
					[
						{
							"type": "paragraph",
							"label": None,
							"subunits":
								[
									{
										"type": "sentence",
										"label": None,
										"text": "A sentence from chapter 2 paragraph 1"
									}
								]
						}
					]
			 }
		]
}
```

## Training data objects
Data contents, labels, and embeddings can be matched using `source` and `key`.
### Data Content
```json
// data content --> output 1 of 2 of `process_structured_text.py`
{
	"metadata": doc["metadata"],
	"content": flat_text, # all base units aggregated in one list
	"key": doc_num, // the number of the doc in its general structure format
	# Optional:
	"structure": structure, # general unified structure; see above
}

```
### Labels
```json
// labels --> output 2 of 2 of `process_structured_text.py`
{
	"base_unit": "sentences", // hierarchy_order[0]
	"labels": {
		"chapters": [1, 0, 0, 0, ...],
		"paragraphs": [1, 0, 1, 0, ...],
	},
	"hierarchy_order": ["sentences", "paragraphs", "chapters"],
	"source": "chicago", // doc["metadata"]["source"],
	"key": doc_num // the number of the doc in its general structure format
}
```

### Embedding
```json
// embeddings --> output of `encode_data.py`
{
	"embeddings": base_unit_embs, // list of base unit embeddings
	"source": "chicago", // or "project_gutenberg", "chapterbreak"
	"key": doc_num // based on data_contents/labels
}
```

