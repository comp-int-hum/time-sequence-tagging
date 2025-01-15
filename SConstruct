import os
import os.path
from steamroller import Environment

vars = Variables("custom.py")

# Add Grid Variables
vars.AddVariables(
    ("STEAMROLLER_ENGINE", "", "slurm"),
    ("GRID_MEMORY", "", "64G"),
    ("GPU_COUNT", "", 1),
    ("GPU_QUEUE", "", ""),
    ("GPU_ACCOUNT", "", None),
    ("CPU_COUNT", "", 2),
    ("CPU_QUEUE", "", "parallel"),
    ("CPU_ACCOUNT", "", None)
)

# General Experiment Variables
vars.AddVariables(
    ("FOLDS", "Number of times to replicate", 1),
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("WW_DATAPATH", "", "/export/data/english/women_writers.tgz"), # correct
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("CHICAGO_PATH", "", "${DATA_ROOT}/us_novels_pre1927.zip"),
    ("TOY_PATH", "", "/work/toy_datasets/"),
    ("LOCAL_DATA", "", ""),
    ("GUTENBERG_CATALOG", "", "pg_catalog.csv"),)

vars.AddVariables(
    ("CHAPTER_FILTERS", "", ["'^chapter\s+\w+\.?$'", "'^.{0,20}$'"]), #  [r"^chapter\s+\w+\.?$", r"^.{0,20}$"]),
    ("ENC_MODEL_NAME", "", "bert-base-uncased"),
    ("ENCODE_SIZE", "Maximum number of texts to encode", 3000),
    ("MAX_TOKS", "", 512),
    ("EMB_DIM", "Size of BERT embedding", 1536),
    ("CLUSTERS", "Number of clusters when grouping", 3),
    ("SPLIT_SIZE", "Maximum number of texts to use during train/test/split", None),
    ("SPLIT_RATIOS", "Ratios to use for train/dev/test", [0.8, 0.1, 0.1]),
    ("SAMPLES", "Number of chapters to sample from text", 5),
    ("EXPERIMENT_TYPE", "", "sequence_tagger"),  # sequence_tagger, cross-classifier, classifier, contrastive
    ("SEQ_LENGTHS", "", [(100, 100)]),
    ("TRAIN_SAMPLES", "How many samples to take from the train text ", [10]),
    (
        "MODELS",
        "Embedding models to try",
        [
            {
                "name" : "bert-tiny",
                "id" : "prajjwal1/bert-tiny"
            }
        ]
    ),
    (
        "MODEL_ARCHITECTURES",
        "Sequence tagging model architectures to try",
        ["lstm"] # "lstm", "minlstm",  mingru
    ),
    ("EPOCHS", "Number of epochs to train model", 30),
    ("EXPERIMENTS", "", []),
    ("TITLE_FILTERS", "", [".*\\bpoe(m|t|s).*", ".*\\bballad.*", ".*\\bverse.*"]),
    ("REQUIRED_PATTERNS", "", [".*fiction.*"]),
    ("LOC_TAGS_TO_KEEP", "Library of Congress tags indicating a text may be kept", ["PS", "PE"]),
)

# Overrides
vars.AddVariables(
    ("GUTENBERG_TEXTS", "", ""),
    ("CHICAGO_TEXTS", "", ""),
    ("ENCODED_TEXTS", "", ""),
    ("SEQ_FILE", "", ""),
    ("TOY_ROOT", "Root of toy dataset directory", f"work/toy_datasets"),
    ("TOY_RUN", "Number of texts in toy run (where 0 is all available texts)", 0),
    ("TOY_FILE", "", "${TOY_ROOT}/toy_${TOY_RUN}.jsonl.gz"),
)

# Variables for strict classifier-based project
vars.AddVariables(
    ("SAME_CH", "", "True"),
    ("CONTEXT_WINDOW_SIZE", "", [2, 10, 30, 50]), # [2, 10, 50]
    ("CD", "", ""), # cd or no_cd
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    BUILDERS={
        "ExtractStructureFromChicago" : Builder(
            action="python scripts/extract_structure_from_chicago.py --chicago_path ${CHICAGO_PATH} --output ${TARGETS[0]} --output_catalog ${TARGETS[1]} --min_chapters ${MIN_CHAPTERS} --chapter_filters ${CHAPTER_FILTERS}",
        ),
        "EncodeData": Builder(
            action="python scripts/encode_data.py --input ${SOURCES[0]} --output ${TARGETS[0]} --model_id ${MODEL_ID} --max_toks ${MAX_TOKS} --batch_size ${BATCH_SIZE}",
        ),
        "GenerateSequence": Builder(
            action="python scripts/generate_sequence.py --input ${SOURCES} --output ${TARGETS[0]} --granularity ${GRANULARITY} --seed ${FOLDS}"
        ),
        "GenerateSequenceHRNN": Builder(
            action="python scripts/generate_sequence_hrnn.py --input ${SOURCES} --output ${TARGETS[0]} --granularity ${GRANULARITY} --seed ${FOLDS}"
        ),
        # "CreateClassifierDatapoints": Builder(
        #     action="python scripts/create_datapoints.py --input ${SOURCES} --output ${TARGETS} --samples ${SAMPLES} --same ${SAME_CH} --context_size ${CONTEXT} --seed ${FOLDS} --min_len ${MIN_LEN}"
        # ),
        "GenerateSplits" : Builder(
            action="python scripts/generate_splits.py --input ${SOURCES[0]} --train ${TARGETS[0]} --dev ${TARGETS[1]} --test ${TARGETS[2]} --random_seed ${RANDOM_SEED} --min_len ${MIN_LEN} --max_len ${MAX_LEN} --sample_method ${SAMPLE_METHOD} --samples_per_document ${SAMPLES_PER_DOCUMENT} --train_proportion ${TRAIN_PROPORTION} --dev_proportion ${DEV_PROPORTION} --test_proportion ${TEST_PROPORTION}"
        ),
        "GenerateDocSplits" : Builder(
            action="python scripts/generate_doc_splits.py --input ${SOURCES[0]} --train ${TARGETS[0]} --dev ${TARGETS[1]} --test ${TARGETS[2]} --random_seed ${RANDOM_SEED} --min_len ${MIN_LEN} --max_len ${MAX_LEN} --sample_method ${SAMPLE_METHOD} --samples_per_document ${SAMPLES_PER_DOCUMENT} --train_proportion ${TRAIN_PROPORTION} --dev_proportion ${DEV_PROPORTION} --test_proportion ${TEST_PROPORTION}"
        ),
        #"SampleSequence": Builder(
        #    action="python scripts/sample_sequence.py --input ${SOURCES} --output ${TARGETS} --min_seq ${MIN_SEQ} --max_seq ${MAX_SEQ} --sample_method ${SAMPLE_METHOD} --samples ${SAMPLES} --seed ${FOLDS} "
        #),
        # "TrainClassifierModel": Builder(
        #     action="python scripts/train_model.py --train ${SOURCES[0]} --test ${SOURCES[1]} --model ${MODEL} --model_name ${SAVE_NAME} --emb_dim ${EMB_DIM} --num_epochs ${EPOCHS} --result ${TARGETS[0]} --errors ${TARGETS[1]} --batch ${BATCH} --confusion ${CM}",
        # ),
        "TrainSequenceModel": Builder(
            action= (
                	"python scripts/train_sequence_model.py "
            		"--train ${SOURCES[0]} "
               		"--dev ${SOURCES[1]} "
                 	"--test ${SOURCES[2]} "
                  	"--output ${TARGETS[0]} "
                    "--model ${ARCHITECTURE} "
                    "--num_epochs ${EPOCHS} "
                    "--batch_size ${BATCH_SIZE} "
                    "--dropout ${DROPOUT} "
                    "--granularity ${GRANULARITY} "
                    "--boundary_type ${BOUNDARY_TYPE}"
			)
        ),
        "TrainHRNN": Builder(
			action= (
       					"python scripts/train_generic_hrnn.py "
                    	"--train ${SOURCES[0]} "
                    	"--dev ${SOURCES[1]} "
                    	"--test ${SOURCES[2]} "
						"--visualization ${VISUALIZATION} "
						"--teacher_ratio ${TEACHER_RATIO} "
                    	"--output ${TARGETS[0]} "
						"--num_epochs ${EPOCHS} "
						"--batch_size ${BATCH_SIZE} "
						"--dropout ${DROPOUT}"
                	)
		)
        # "GenerateReport": Builder(
        #     action="python scripts/generate_report.py --input ${SOURCES[0]} --output ${TARGETS} --pg_path ${PG_PATH}"
        # ),
        # "CollateResults": Builder(
        #     action="python scripts/collate_results.py --data ${SOURCES} --target ${TARGETS}"
        # ),
    }
)

chicago_docs = env.ExtractStructureFromChicago(
    source = [],
    target = ["work/chicago_texts.jsonl.gz", "work/chicago_extracted_texts_catalog.txt.gz"],
    MIN_CHAPTERS = 3
)

for model in env.get("MODELS", []):
    enc_texts = env.EncodeData(
        source = [chicago_docs],
        target = ["work/${MODEL_NAME}/encoded_texts.jsonl.gz"],
        MODEL_NAME=model["name"],
        MODEL_ID=model["id"],
        BATCH_SIZE=model.get("BATCH_SIZE", 64),
        STEAMROLLER_GPU_COUNT=1,
        STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
        STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
        STEAMROLLER_TIME = "12:00:00",
        STEAMROLLER_MEMORY="32G"
    )

    for granularity in ["sentence"]:
        seq_file = env.GenerateSequenceHRNN(
            source = [enc_texts],
            target = ["work/${MODEL_NAME}/${GRANULARITY}/sequence_embeddings.jsonl.gz"],
            MODEL_NAME=model["name"],
            GRANULARITY = granularity
        )
        for fold in range(env["FOLDS"]):
            for min_seq, max_seq in env["SEQ_LENGTHS"]:
                train, dev, test = env.GenerateDocSplits(
                    source=seq_file,
                    target=[
                        "work/${MODEL_NAME}/${GRANULARITY}/train.jsonl.gz",
                        "work/${MODEL_NAME}/${GRANULARITY}/dev.jsonl.gz",
                        "work/${MODEL_NAME}/${GRANULARITY}/test.jsonl.gz"
                    ],
                    MODEL_NAME=model["name"],
                    GRANULARITY = granularity,
                    RANDOM_SEED=fold,
                    MIN_LEN = min_seq,
                    MAX_LEN = max_seq,
                    SAMPLE_METHOD = "from_beginning",
                    SAMPLES_PER_DOCUMENT = 100,
                    TRAIN_PROPORTION = 0.8,
                    DEV_PROPORTION = 0.1,
                    TEST_PROPORTION = 0.1
                )
                result = env.TrainHRNN(
                            source = [train, dev, test],
                            target = ["work/${MODEL_NAME}/model_output.txt"],
                            MODEL_NAME = model["name"],
                            BATCH_SIZE = 1024,
                            DROPOUT = 0.6,
                            VISUALIZATION = f"work",
                            TEACHER_RATIO = 0.8,
                            EPOCHS = 20,
                            STEAMROLLER_GPU_COUNT=1,
							STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
							STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
							STEAMROLLER_TIME = "12:00:00",
							STEAMROLLER_MEMORY="32G"
                        )
                
                # for boundary_type in ["chapter", "paragraph", "both"]:
                #     for architecture in env["MODEL_ARCHITECTURES"]:
                #         result = env.TrainSequenceModel(
                #             source = [train, dev, test],
                #             target = ["work/${MODEL_NAME}/${GRANULARITY}/${ARCHITECTURE}/${BOUNDARY_TYPE}/model.bin"],
                #             MODEL_NAME = model["name"],
                #             ARCHITECTURE=architecture,
                #             GRANULARITY=granularity,
                #             BOUNDARY_TYPE=boundary_type,
                #             BATCH_SIZE=1024,
                #             DROPOUT = 0.6
                #         )
                
