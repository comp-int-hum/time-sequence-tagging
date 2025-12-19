from bdb import Breakpoint
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
    ("TRAIN_SOURCE", "", "chicago"),
    ("DATASETS_TO_PROCESS", "", ["chicago", "chapterbreak"])
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
    ("ARCHITECTURE", "", ["RNN", "HRNN"])
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
    ("HRNN_LAYER_NAMES", "Hierarchical level names", ["sentences", "paragraphs", "chapters"]),
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

        "LogTest" : Builder(
            action= ("python scripts/log_test.py "
                     "--output ${TARGETS[0]} "
            )
        ),

        # 1. DATASET BUILDERS

        # a) chicago
        "ExtractStructureFromChicago" : Builder(
            action= ("python scripts/data/datasets/chicago/extract_structure_from_chicago.py "
                     "--chicago_path ${CHICAGO_PATH} "
                     "--output ${TARGETS[0]} "
                     "--output_catalog ${TARGETS[1]} "
                     "--min_chapters ${MIN_CHAPTERS} "
                     "--chapter_heading_patterns ${CHAPTER_HEADING_PATTERNS}"
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
                      "--data_output ${TARGETS[0]} "
                      "--label_output ${TARGETS[1]} "
                      "--splits ${SPLITS} "
                      '--title_filters "${SENTENCE_FILTERS}"'
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
                      "--data_output ${TARGETS[0]} "
                      "--label_output ${TARGETS[1]}"
            )
        ),
        # 4. ENCODE BUILDERS

        "EncodeData": Builder(
            action= ("python scripts/data/encode_data.py "
                     "--input ${SOURCES[0]} "
                     "--output_embedding ${TARGETS[0]} "
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
                "${'--random_seed %s' % RANDOM_SEED if RANDOM_SEED else ''}"
                # "${RANDOM_SEED and '--random_seed' ${RANDOM_SEED} or ''}"
            )
        ),
        
        # 6. TRAIN HRNN
        "TrainHRNN": Builder(
            action=(
                "python scripts/training/train_hrnn.py "
                "--train ${SOURCES[0]} ${SOURCES[1]} ${SOURCES[2]} "
                "--dev ${SOURCES[3]} ${SOURCES[4]} ${SOURCES[5]} "
                "--architecture ${ARCHITECTURE} "
                "--train_output ${TARGETS[0]} "
                "--dev_output ${TARGETS[1]} "
                "--train_stats ${TARGETS[2]} "
                "--train_dump ${TARGETS[3]} "
                "--trained_model ${TARGETS[4]} "
                "--num_epochs ${NUM_EPOCHS} "
                "--batch_size ${BATCH_SIZE} "
                "--dropout ${DROPOUT} "
                "--threshold ${THRESHOLD} "
                "--temperature ${TEMPERATURE} "
                "--hrnn_layer_names ${HRNN_LAYER_NAMES} "
                "${LAYER_LOSS_WEIGHTS and '--layer_loss_weights ' + ' '.join(map(str, LAYER_LOSS_WEIGHTS)) or ''} "
                "${BALANCE_POS_NEG and '--balance_pos_neg ' + ' '.join(map(str, BALANCE_POS_NEG)) or ''} "
                "${TEACHER_RATIO and '--teacher_ratio ' + str(TEACHER_RATIO) or ''}"
            )
        ),
        # Plot metrics
        "PlotTrainingMetrics": Builder(
            action=(
                "python scripts/evaluation/metrics/plot_training_metrics.py "
                "--input ${SOURCES[0]} "
                "--loss_curves ${TARGETS[0]} "
                "--layer_loss_curves ${TARGETS[1:3]} "
                # "${HRNN_LAYER_NAMES and '--hrnn_layer_names' + ' '.join(HRNN_LAYER_NAMES) or ''}"
            )
        ),
        "ComputeROCMetrics": Builder(
            action=(
                "python scripts/evaluation/metrics/compute_roc_metrics.py "
                "--input ${SOURCES[0]} "
                "--optimal_thresholds_output ${TARGETS[0]} "
                "--roc_by_layer ${TARGETS[1:]} "
                # "${HRNN_LAYER_NAMES and '--hrnn_layer_names' + ' '.join(HRNN_LAYER_NAMES) or ''}"
            )
        ),
        "ComputeConfidenceMetrics": Builder(
            action=(
                "python scripts/evaluation/metrics/compute_text_confidence_metrics.py "
                "--input ${SOURCES[0]} "
                "--confidence_matrix ${TARGETS[0]} "
                # "${HRNN_LAYER_NAMES and '--hrnn_layer_names' + ' '.join(HRNN_LAYER_NAMES) or ''} "
                "--threshold ${THRESHOLD}"
            )
        ),
        # Utils
        "TruncateFile": Builder(
            action=(
                "python utils/truncate.py "
                "--input ${SOURCES[0]} "
                "--output ${TARGETS[0]} "
                "--retain_lines ${RETAIN_LINES}"
            )
        ),

    }
)

env["ENV"]["PYTHONPATH"] = os.getcwd()

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


env.LogTest(source = [],
            target = ["output.txt"])

# Datasets
processed_datasets = {}

# Process structured chicago docs
if "chicago" in env.get("DATASETS_TO_PROCESS", []):
    structured_chicago_docs = env.ExtractStructureFromChicago(
        source = [],
        target = [
            "work/datasets/chicago/structured_chicago_texts.jsonl.gz",
            "work/datasets/chicago/extracted_chicago_texts_catalog.txt.gz"
        ],
        MIN_CHAPTERS = 3,
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

    processed_datasets["chicago"] = {
        "data": chicago_data,
        "labels": chicago_labels,
    }

# Process chapterbreak docs into data, labels, and embeddings
if "chapterbreak" in env.get("DATASETS_TO_PROCESS", []):
    chapterbreak_data, chapterbreak_labels = env.RestructureChapterbreak(
        source = [env.get("CHAPTERBREAK_FILE")],
        target = [f"work/datasets/chapterbreak/chapterbreak_data.jsonl.gz",
                f"work/datasets/chapterbreak/chapterbreak_hierarchical_labels.jsonl.gz"],
        SPLITS = ["pg19", "ao3"],
        FILTERS = env["SENTENCE_FILTERS"]
    )

    processed_datasets["chapterbreak"] = {
        "data": chapterbreak_data,
        "labels": chapterbreak_labels,
    }

# Process gutenberg docs
if "gutenberg" in env.get("DATASETS_TO_PROCESS", []):
    print("gutenberg not ready")

    # gutenberg_docs = env.FetchFromGutenberg(
    #     source = [],
    #     target = [f"work/datasets/gutenberg/gutenberg_docs.jsonl.gz"],
    # )


TRAIN_SOURCE = env["TRAIN_SOURCE"]
TOY_RUN = int(env.get("TOY_RUN", 0))

if TRAIN_SOURCE not in processed_datasets:
    raise KeyError(f"Train source '{env['TRAIN_SOURCE']}' has not been processed. "
                   f"Available datasets: {list(processed_datasets.keys())}")

train_dataset = processed_datasets[TRAIN_SOURCE]

if TOY_RUN > 0:
    # Truncate and then encode
    toy_root = f"{env['TOY_ROOT']}/{TRAIN_SOURCE}"
    
    toy_data = env.TruncateFile(
        source=[train_dataset["data"]],
        target=[f"{toy_root}/{TRAIN_SOURCE}/{TOY_RUN}_data.jsonl.gz"],
        RETAIN_LINES=TOY_RUN
    )

    toy_labels = env.TruncateFile(
        source=[train_dataset["labels"]],
        target=[f"{toy_root}/{TRAIN_SOURCE}/{TOY_RUN}_labels.jsonl.gz"],
        RETAIN_LINES=TOY_RUN
    )

    toy_embeddings = env.EncodeData(
        source=[toy_data[0]],
        target=[f"{toy_root}/{TRAIN_SOURCE}/{TOY_RUN}_embeddings.jsonl.gz"],
        MODEL_ID=env["MODELS"][0]["id"],
        MAX_TOKS=env["MAX_TOKS"],
        BATCH_SIZE=env.get("BATCH_SIZE", 512)
    )

    train_dataset["toyset"] = {
        "data": toy_data[0],
        "labels": toy_labels[0],
        "embeddings": toy_embeddings[0]
    }

else:
    # Full encode datasets
    for dname, dataset in processed_datasets.items():
        dataset["embeddings"] = env.EncodeData(
            source = [dataset["data"]],
            target = [f"work/datasets/{dname}/{dname}_embeddings.jsonl.gz"],
            MODEL_ID = env["MODELS"][0]["id"],
            MAX_TOKS = env["MAX_TOKS"],
            BATCH_SIZE = 512
        )

if TOY_RUN > 0:
    split_input = train_dataset["toyset"]
    split_prefix = f"work/experiments/{TRAIN_SOURCE}/toy_{TOY_RUN}"
else:
    split_input = train_dataset
    split_prefix = f"work/experiments/{TRAIN_SOURCE}/full"

train_source_splits = env.GenerateSplits(
    source = [
        split_input["data"],
        split_input["labels"],
        split_input["embeddings"]
    ],
    target = [
        f"{split_prefix}/data.train.gz",
        f"{split_prefix}/labels.train.gz",
        f"{split_prefix}/embeddings.train.gz",

        f"{split_prefix}/data.dev.gz",
        f"{split_prefix}/labels.dev.gz",
        f"{split_prefix}/embeddings.dev.gz",

        f"{split_prefix}/data.test.gz",
        f"{split_prefix}/labels.test.gz",
        f"{split_prefix}/embeddings.test.gz",
    ],
    TRAIN_PROPORTION = env["SPLIT_RATIOS"][0],
    DEV_PROPORTION = env["SPLIT_RATIOS"][1],
    TEST_PROPORTION = env["SPLIT_RATIOS"][2],
    RANDOM_SEED = env.get("RANDOM_SEED", 0),
    **cpu_task_config(f"splits_{TRAIN_SOURCE}", "01:00:00", "32G")
)

architecture_results = {}
for architecture in env.get("ARCHITECTURE", []):
    train_guesses, dev_guesses, training_summary, training_metrics, trained_model = env.TrainHRNN(
        source = [train_source_splits[:6]],
        target = [f"{split_prefix}/{architecture}/guesses/train_guesses.pkl",
                    f"{split_prefix}/{architecture}/guesses/dev_guesses.pkl",
                    f"{split_prefix}/{architecture}/results/train_stats.txt",
                    f"{split_prefix}/{architecture}/results/train_dump.pkl",
                    f"{split_prefix}/{architecture}/model/model_state.pth"],
        BATCH_SIZE = 4,
        ARCHITECTURE = "HRNN",
        DROPOUT = 0.6,
        TEACHER_RATIO = 1.0,
        NUM_EPOCHS = env.get("EPOCHS"),
        THRESHOLD =  env["THRESHOLD"],
        BALANCE_POS_NEG = [1.0, 1.0],
        TEMPERATURE = 1.0,
        # LAYER_WEIGHTS = [1.0, layer_weight],
        **gpu_task_config("train_hrnn", "12:00:00", "32G"),
    )
    architecture_results[architecture] = {
        "train_guesses": train_guesses,
        "dev_guesses": dev_guesses,
        "train_summary": training_summary,
        "train_metrics": training_metrics,
        "trained_model": trained_model,
    }

for architecture, results in architecture_results.items():
    # Extract outputs from training
    training_metrics = results["train_metrics"]
    dev_guesses = results["dev_guesses"]

    # Architecture-specific paths
    arch_results_dir = f"{split_prefix}/{architecture}/results"
    arch_guesses_dir = f"{split_prefix}/{architecture}/guesses"
    arch_roc_dir = f"{split_prefix}/{architecture}/roc"
    arch_conf_dir = f"{split_prefix}/{architecture}/confidence"

    # Plot training metrics
    training_metrics_visualizations = env.PlotTrainingMetrics(
        source=[training_metrics],
        target=[
            f"{arch_results_dir}/loss_curves.png",
            f"{arch_results_dir}/par_layer_loss_curves.png",
            f"{arch_results_dir}/chapter_layer_loss_curves.png",
        ]
    )

    # ROC visualizations (one per layer)
    roc_visualizations = [
        f"{arch_roc_dir}/roc_curve_for_{layer_name}.png"
        for layer_name in env["HRNN_LAYER_NAMES"][1:]
    ]

    roc_metrics = env.ComputeROCMetrics(
        source=[dev_guesses],
        target=[
            f"{arch_roc_dir}/optimal_thresholds.json",
            *roc_visualizations  # unpack image paths
        ],
    )

    # Confidence metrics
    confidence_matrix = env.ComputeConfidenceMetrics(
        source=[dev_guesses],
        target=[f"{arch_conf_dir}/confidence_matrix.json"],
        THRESHOLD=env["THRESHOLD"],
    )