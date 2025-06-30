import os
import os.path
from steamroller import Environment
import json
import logging
from utils.log_setup import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

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
    ("WORK_DIR", "", "work"),
    ("LOCAL_DATA", "", ""),
)

# Datasets
vars.AddVariables(
    ("GUTENBERG_CATALOG", "", "pg_catalog.csv"),
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("CHICAGO_PATH", "", "${DATA_ROOT}/us_novels_pre1927.zip"),
    ("ARXIV_FILE", "", "/home/sli159/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/218/arxiv-metadata-oai-snapshot.json"),
    ("CHAPTERBREAK_FILE", "", "${DATA_ROOT}/chapterbreak/chapterbreak_ctx_1024.json"),
    ("WOMEN_WRITERS_PATH", "", "/export/data/english/women_writers.tgz"), # correct
)

vars.AddVariables(
    ("TOY_ROOT", "", "${WORK_DIR}/toy_datasets/"),
    ("TOY_RUN", "Number of texts to include in toy run (where 0 is all available texts)", 0),
    ("TOY_FILE", "", "${TOY_ROOT}/toy_${TOY_RUN}.jsonl.gz")
)

# regexes
vars.AddVariables(
    ("CHAPTER_HEADING_PATTERNS", "", ["'^chapter\s+\w+\.?$'", "'^.{0,20}$'"]), # for chicago
    ("TITLE_FILTERS", "", [".*\\bpoe(m|t|s).*", ".*\\bballad.*", ".*\\bverse.*"]), # for gutenberg
    ("SENTENCE_FILTERS", "", [r"^[^a-zA-Z0-9]*[A-Z]+(?:'[A-Z]+)?(?:\s[A-Z]+(?:'[A-Z]+)?)*[^a-zA-Z0-9]*$"]), # for gutenberg, chapterbreak
    ("REQUIRED_PATTERNS", "", [".*fiction.*"]), # for gutenberg
)

vars.AddVariables(
    ("ENC_MODEL_NAME", "", "bert-base-uncased"),
    ("ENCODE_SIZE", "Maximum number of texts to encode", 3000),
    ("MAX_TOKS", "", 512),
    ("EMB_DIM", "Size of BERT embedding", 1536),
    ("CLUSTERS", "Number of clusters when grouping", 3),
    ("SPLIT_SIZE", "Maximum number of texts to use during train/test/split", None),
    ("SPLIT_RATIOS", "Ratios to use for train/dev/test", [0.8, 0.1, 0.1]),
    ("SAMPLES", "Number of chapters to sample from text", 5),
    ("EXPERIMENT_TYPE", "", "sequence_tagger"),  # sequence_tagger, cross-classifier, classifier, contrastive
    ("SEQ_LENGTHS", "", [(10000, 20000)]),
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
    ("LOC_TAGS_TO_KEEP", "Library of Congress tags indicating a text may be kept", ["PS", "PE"]),
    ("LAYER_WEIGHTS", "", [1.0]),
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

vars.AddVariables(
    ("HRNN_LAYER_NAMES", "Hierarchical level names", ["paragraphs", "chapters"]),
    ("VIS_NUM", "Visualization Number", 4)
)

# Inference time
vars.AddVariables(
	("THRESHOLD", "Prediction threshold for hierarchical boundaries", 0.5),
 	("TEMPERATURE", "Temperature value for hidden state mixture", 0.8)
)

dataset_builders = {
}

env = Environment(
    variables=vars,
    # ENV=os.environ,
    BUILDERS={
        # 1. DATASET BUILDERS

        # a) chicago
        "ExtractStructureFromChicago" : Builder(
            action= ("python scripts/data/datasets/chicago/extract_structure_from_chicago.py "
                     "--chicago_path ${CHICAGO_PATH} "
                     "--output ${TARGETS[0]} "
                     "--output_catalog ${TARGETS[1]} "
                     "--min_chapters ${MIN_CHAPTERS} "
                     "--chapter_filters ${CHAPTER_HEADING_PATTERNS}"
            )
        ),

        # b) gutenberg
        "FetchFromGutenberg": Builder(
            action = ("python scripts/data/datasets/gutenberg/fetch_from_gutenberg.py "
                      "--gutenberg_path ${GUTENBERG_PATH} "
                      "--output ${TARGETS[0]} "
                      "--title_filters ${' '.join(['\"' + x + '\"' for x in TITLE_FILTERS])} "
                      "--loc_tags_to_keep ${LOC_TAGS_TO_KEEP} "
                      "--must_occur ${' '.join(['\"' + x + '\"' for x in REQUIRED_PATTERNS])}"
            )
        ),
        "ExtractStructureFromGutenberg": Builder(
            action = ("python scripts/data/datasets/gutenberg/extract_structure_from_gutenberg.py "
                      "--input ${SOURCES[0]} "
                      "--output ${TARGETS[0]} "
                      "${LIMIT and '--limit' ${LIMIT} or ''}"
            )
        ),

        # c) chapterbreak
        "RestructureChapterbreak": Builder(
            action = ("python scripts/data/datasets/chapterbreak/restructure_chapterbreak.py "
                      "--input ${SOURCES[0]} "
                      "--output ${TARGETS[0]} "
                      "--splits ${SPLITS} "
                      "--title_filters ${SENTENCE_FILTERS}"
            )
        ),

        # 2. STRUCTURED DATA CLEANER
        "CleanStructuredData": Builder(
            action = ('python scripts/data/datasets/text_structure_cleaner.py '
                     '--input ${SOURCES[0]} '
                     '--output ${TARGETS[0]} '
                     '--filters "${FILTERS}" '
                     '--min_chapter_len ${MIN_CH_LEN} '
                     "${MERGE_PARAGRAPHS and '--merge_paragraphs' or ''}"
            )	
        ),

        # 3. PROCESS STRUCTURED TEXT (if needed)
        "ProcessStructuredText": Builder(
            action = ("python scripts/data/datasets/process_structured_text.py "
                      "--input ${SOURCES[0]} "
                      "--data_output ${TARGETS[0]}"
                      "--label_output ${TARGETS[1]}"
            )
        ),
        # 4. ENCODE BUILDERS

        "EncodeData": Builder(
            action= ("python scripts/data/encode_data.py "
                     "--input ${SOURCES[0]} "
                     "--output_embeddings ${TARGETS[0]} "
                     "--model_id ${MODEL_ID} "
                     "--max_toks ${MAX_TOKS} "
                     "--batch_size ${BATCH_SIZE}"
            )
        ),

        # 5. GENERATE SPLITS
        "GenerateSplits": Builder(
            action=(
                "python scripts/data/generate_splits.py "
                "--data_input ${SOURCES[0]} "
                "--label_input ${SOURCES[1]} "
                "--embedding_input ${SOURCES[2]} "
                "--train ${TARGETS[0]} ${TARGETS[1]} ${TARGETS[2]} "
                "--dev ${TARGETS[3]} ${TARGETS[4]} ${TARGETS[5]} "
                "--test ${TARGETS[6]} ${TARGETS[7]} ${TARGETS[8]} "
                "--train_proportion ${TRAIN_PROPORTION} "
                "--dev_proportion ${DEV_PROPORTION} "
                "--test_proportion ${TEST_PROPORTION} "
                "${RANDOM_SEED and '--random_seed ${RANDOM_SEED}' or ''}"
            )
        )

    }
)

def cpu_task_config(name, time_required, memory_required=env["GRID_MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["CPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["CPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
    }

def gpu_task_config(name, time_required, memory_required=env["GRID_MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["GPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["GPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
        "STEAMROLLER_GPU_COUNT": env["GPU_COUNT"],
    }

# Datasets

datasets = {}

# Process gutenberg docs

# gutenberg_docs = env.FetchFromGutenberg(
#     source = [],
#     target = [f"work/datasets/gutenberg/gutenberg_docs.jsonl.gz"],
# )

# Process structured chicago docs
structured_chicago_docs = env.ExtractStructureFromChicago(
    source=[],
    target=[
        "work/datasets/chicago/structured_chicago_texts.jsonl.gz",
        "work/datasets/chicago/extracted_chicago_texts_catalog.txt.gz"
    ],
    MIN_CHAPTERS=3
)

cleaned_chicago_docs = env.CleanStructuredData(
    source = [structured_chicago_docs],
    target = ["work/datasets/chicago/cleaned_structured_chicago_texts.jsonl.gz"],
    FILTERS = env["SENTENCE_FILTERS"],
    MIN_CH_LEN = 2,
    MERGE_PARAGRAPHS = True,
    **cpu_task_config("clean_chicago_docs", "12:00:00", "64G")
)

# Convert to data, labels, and embeddings
chicago_data, chicago_labels = env.ProcessStructuredText(
    source = [cleaned_chicago_docs],
    target = ["work/datasets/chicago/chicago_data.jsonl.gz",
              "work/datasets/chicago/chicago_hierachical_data.jsonl.gz"],
)

chicago_embeddings = env.EncodeData(
    source = [chicago_data],
    target = [f"work/datasets/chicago/chicago_embeddings.jsonl.gz"],
    MODEL_ID = env["MODELS"][0]["id"],
    MAX_TOKS = env["MAX_TOKS"],
    BATCH_SIZE = 512
)

# Process chapterbreak docs into data, labels, and embeddings
chapterbreak_data, chapterbreak_labels = env.RestructureChapterbreak(
    source = [env.get("CHAPTERBREAK_FILE")],
    target = [f"work/datasets/chapterbreak/chapterbreak_data.jsonl.gz",
              f"work/datasets/chapterbreak/chapterbreak_hierarchical_labels.jsonl.gz"],
    SPLITS = ["pg19", "ao3"],
    FILTERS = repr(env["SENTENCE_FILTERS"])
)

chapterbreak_embeddings = env.EncodeData(
    source = [chapterbreak_data],
    target = [f"work/datasets/chapterbreak/chapterbreak_embeddings.jsonl.gz"],
    MODEL_ID = env["MODELS"][0]["id"],
    MAX_TOKS = env["MAX_TOKS"],
    BATCH_SIZE = 512
)

# Split data for chapterbreak
env.GenerateSplits(
    source = [chapterbreak_data, chapterbreak_labels, chapterbreak_embeddings],
    target = [
        # train
        "chapterbreak_data.train.gz",
        "chapterbreak_labels.train.gz",
        "chapterbreak_embeddings.train.gz",

        # dev
        "chapterbreak_data.dev.gz",
        "chapterbreak_labels.dev.gz",
        "chapterbreak_embeddings.dev.gz",

        # test
        "chapterbreak_data.test.gz",
        "chapterbreak_labels.test.gz",
        "chapterbreak_embeddings.test.gz",
    ],
    TRAIN_PROPORTION = env["SPLIT_RATIOS"][0],
    DEV_PROPORTION = env["SPLIT_RATIOS"][1],
    TEST_PROPORTION = env["SPLIT_RATIOS"][2],
    RANDOM_SEED = 0
)

# Split data for Chicago
env.GenerateSplits(
    source = [chicago_data, chicago_labels, chicago_embeddings],
    target = [
        # train
        "work/datasets/chicago/chicago_data.train.gz",
        "work/datasets/chicago/chicago_labels.train.gz",
        "work/datasets/chicago/chicago_embeddings.train.gz",

        # dev
        "work/datasets/chicago/chicago_data.dev.gz",
        "work/datasets/chicago/chicago_labels.dev.gz",
        "work/datasets/chicago/chicago_embeddings.dev.gz",

        # test
        "work/datasets/chicago/chicago_data.test.gz",
        "work/datasets/chicago/chicago_labels.test.gz",
        "work/datasets/chicago/chicago_embeddings.test.gz",
    ],
    TRAIN_PROPORTION = env["SPLIT_RATIOS"][0],
    DEV_PROPORTION = env["SPLIT_RATIOS"][1],
    TEST_PROPORTION = env["SPLIT_RATIOS"][2],
    RANDOM_SEED = 0
)
        
# for model in env.get("MODELS", []):
#     enc_texts = env.EncodeData(
#         source = [transformed_chicago_docs],
#         target = ["work/${MODEL_NAME}/encoded_texts.jsonl.gz"],
#         MODEL_NAME=model["name"],
#         MODEL_ID=model["id"],
#         BATCH_SIZE=model.get("BATCH_SIZE", 64),
#         **gpu_task_config("encode_data", "12:00:00", "32G"),
#     )
    
#     encoded_chapterbreak_data = env.EncodeSequencedTexts(
#         source = [chapterbreak_data],
#         target = ["work/${MODEL_NAME}/encoded_chapterbreak.jsonl.gz"],
#         MODEL_NAME=model["name"],
#         MODEL_ID=model["id"],
#         BATCH_SIZE=model.get("BATCH_SIZE", 64),
#         **gpu_task_config("encode_data", "12:00:00", "32G"),
#     )
    
#     if env.get("TOY_RUN", 0):
#         print(f"Toy run: {env['TOY_RUN']}")
#         if env.get("TOY_FILE", None):
#             enc_texts = env.TruncateData(
#                 source = [enc_texts],
#                 target = [env.get("TOY_FILE")],
#                 RETAIN_LINES = env["TOY_RUN"]
#             )
#         else:
#             enc_texts = env.File(env["TOY_RUN"])

#     seq_file = env.GenerateSequenceHRNN(
#         source = [enc_texts],
#         target = ["work/${MODEL_NAME}/sequence_embeddings.jsonl.gz"],
#         MODEL_NAME=model["name"],
#     )
    # for fold in range(env["FOLDS"]):
    #     for min_seq, max_seq in env["SEQ_LENGTHS"]:
    #         train, dev, test = env.GenerateDocSplits(
    #             source=seq_file,
    #             target=[
    #                 "work/${MODEL_NAME}/train.jsonl.gz",
    #                 "work/${MODEL_NAME}/dev.jsonl.gz",
    #                 "work/${MODEL_NAME}/test.jsonl.gz"
    #             ],
    #             MODEL_NAME=model["name"],
    #             RANDOM_SEED=fold,
    #             MIN_LEN = min_seq,
    #             MAX_LEN = max_seq,
    #             SAMPLE_METHOD = "from_beginning",
    #             SAMPLES_PER_DOCUMENT = 1,
    #             TRAIN_PROPORTION = 0.8,
    #             DEV_PROPORTION = 0.1,
    #             TEST_PROPORTION = 0.1
    #         )
            
    #         for layer_weight in env.get("LAYER_WEIGHTS", []):
    #             layer_weight_path = f"{env['WORK_DIR']}/{model['name']}/trained_model_output/layer_weight_{layer_weight}"
    #             visualization_path = f"{layer_weight_path}/visualizations"
    #             roc_path = f"{layer_weight_path}/roc_metrics"
                
    #             train_guesses, dev_guesses, training_summary, training_metrics, trained_model = env.TrainHRNN(
    #                 source = [train, dev, test],
    #                 target = [f"{layer_weight_path}/guesses/train_guesses.pkl",
    #                             f"{layer_weight_path}/guesses/dev_guesses.pkl",
    #                             f"{layer_weight_path}/results/training_results.txt",
    #                             f"{layer_weight_path}/results/training_metrics.pkl",
    #                             f"{layer_weight_path}/model/model_state.pth"],
    #                 MODEL_NAME = model["name"],
    #                 BATCH_SIZE = 10,
    #                 DROPOUT = 0.6,
    #                 TEACHER_RATIO = 1.0,
    #                 EPOCHS = env.get("EPOCHS"),
    #                 THRESHOLD =  env["THRESHOLD"],
    #                 BALANCE_POS_NEG = [1.0, 1.0],
    #                 LAYER_WEIGHTS = [1.0, layer_weight],
    #                 **gpu_task_config("train_hrnn", "12:00:00", "32G"),
    #             )
                
    #             training_metrics_visualizations = env.PlotTrainingMetrics(
    #                 source = [training_metrics],
    #                 target = [
    #                     f"{layer_weight_path}/results/loss_curves.png",
    #                     f"{layer_weight_path}/results/par_layer_loss_curves.png",
    #                     f"{layer_weight_path}/results/chapter_layer_loss_curves.png",
    #                 ]
    #             )
                
    #             # Get ROC metrics and visualizations
    #             roc_visualizations = [f"{roc_path}/roc_curve_for_{layer_name}" for layer_name in env["HRNN_LAYER_NAMES"]]
                
    #             roc_metrics = env.ComputeROCMetrics(
    #                 source = [dev_guesses],
    #                 target = [f"{roc_path}/optimal_thresholds.json",
    #                             roc_visualizations],
    #             )
                
    #             # Get confidence matrix
    #             confidence_matrix = env.ComputeConfidenceMetrics(
    #                 source = [dev_guesses],
    #                 target = [f"{layer_weight_path}/confidence_matrix.json"],
    #                 THRESHOLD = env["THRESHOLD"],
    #             )
                
    #             # Get boundary visualizations
    #             layer_visualizations = [f"{visualization_path}/visualization_{num}" for num in range(env["VIS_NUM"])]
                
    #             boundary_visualizations = env.BuildVisualizations(
    #                 source = [dev_guesses],
    #                 target = [layer_visualizations],
    #                 THRESHOLD = env["THRESHOLD"],
    #                 NUM_LAYERS = len(env["HRNN_LAYER_NAMES"])
    #             )
                
    #             chapterbreak_guesses = env.Evaluate(
    #                 source = [encoded_chapterbreak_data, trained_model],
    #                 target = [f"{layer_weight_path}/chapterbreak_guesses.pkl"],
    #                 BATCH_SIZE = 10,
    #                 THRESHOLD =  env["THRESHOLD"]
    #             )