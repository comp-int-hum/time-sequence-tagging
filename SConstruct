import os
import os.path
from steamroller import Environment
import json

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
    ("ARXIV_FILE", "", "/home/sli159/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/218/arxiv-metadata-oai-snapshot.json"),
    ("CHAPTERBREAK_FILE", "", "${DATA_ROOT}/chapterbreak/chapterbreak_ctx_1024.json"),
    ("WORK_DIR", "", "work"),
    ("WW_DATAPATH", "", "/export/data/english/women_writers.tgz"), # correct
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("CHICAGO_PATH", "", "${DATA_ROOT}/us_novels_pre1927.zip"),
    ("TOY_PATH", "", "/work/toy_datasets/"),
    ("LOCAL_DATA", "", ""),
    ("GUTENBERG_CATALOG", "", "pg_catalog.csv"),
)

vars.AddVariables(
    ("TOY_ROOT", "", "${WORK_DIR}/toy_datasets/"),
    ("TOY_RUN", "Number of texts to include in toy run (where 0 is all available texts)", 0),
    ("TOY_FILE", "", "${TOY_ROOT}/toy_${TOY_RUN}.jsonl.gz")
)

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
    ("TITLE_FILTERS", "", [".*\\bpoe(m|t|s).*", ".*\\bballad.*", ".*\\bverse.*"]),
    ("SENTENCE_FILTERS", "", [r"^[^a-zA-Z0-9]*[A-Z]+(?:'[A-Z]+)?(?:\s[A-Z]+(?:'[A-Z]+)?)*[^a-zA-Z0-9]*$"]),
    ("REQUIRED_PATTERNS", "", [".*fiction.*"]),
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

env = Environment(
    variables=vars,
    ENV=os.environ,
    BUILDERS={
        "ExtractStructureFromChicago" : Builder(
            action= ("python scripts/extract_structure_from_chicago.py "
                     "--chicago_path ${CHICAGO_PATH} "
                     "--output ${TARGETS[0]} "
                     "--output_catalog ${TARGETS[1]} "
                     "--min_chapters ${MIN_CHAPTERS} "
                     "--chapter_filters ${CHAPTER_FILTERS}"
            )
        ),
        "TransformExtractedData": Builder(
            action= ('python scripts/transform_extracted_data.py '
                     '--input ${SOURCES[0]} '
                     '--output ${TARGETS[0]} '
                     '--filters "${FILTERS}" '
                     '--min_avg_chapter_len ${MIN_AVG_CH_LEN} '
                     "${MERGE_PARAGRAPHS and '--merge_paragraphs' or ''}"
            )	
        ),
        "TransformEncodedData": Builder(
            action= ('python scripts/transform_encoded_data.py '
                     '--input ${SOURCES[0]} '
                     '--output ${TARGETS[0]} '
                     '--filters "${FILTERS}" '
                     '--min_avg_chapter_len ${MIN_AVG_CH_LEN} '
                     "${MERGE_PARAGRAPHS and '--merge_paragraphs' or ''}"
            )	
        ),
        "EncodeData": Builder(
            action= ("python scripts/encode_data.py "
                     "--input ${SOURCES[0]} "
                     "--output ${TARGETS[0]} "
                     "--model_id ${MODEL_ID} "
                     "--max_toks ${MAX_TOKS} "
                     "--batch_size ${BATCH_SIZE}"
            )
        ),
        
        "TruncateData": Builder(
            action= ("python scripts/truncate.py "
                     "--input ${SOURCES[0]} "
                     "--output ${TARGETS[0]} "
                     "--retain_lines ${RETAIN_LINES}"
             )
        ),
        # "GenerateSequence": Builder(
        #     action= ("python scripts/generate_sequence.py "
        #              "--input ${SOURCES} "
        #              "--output ${TARGETS[0]} "
        #              "--granularity ${GRANULARITY} "
        #              "--seed ${FOLDS}"
        # 	)
        # ),
        "GenerateSequenceHRNN": Builder(
            action = ("python scripts/generate_sequence_hrnn.py "
                      "--input ${SOURCES} "
                      "--output ${TARGETS[0]} "
                      "--granularity ${GRANULARITY} "
                      "--seed ${FOLDS}"
            )
        ),
        # # "CreateClassifierDatapoints": Builder(
        # #     action="python scripts/create_datapoints.py --input ${SOURCES} --output ${TARGETS} --samples ${SAMPLES} --same ${SAME_CH} --context_size ${CONTEXT} --seed ${FOLDS} --min_len ${MIN_LEN}"
        # # ),
        # "GenerateSplits" : Builder(
        #     action= ("python scripts/generate_splits.py "
        #              "--input ${SOURCES[0]} "
        #              "--train ${TARGETS[0]} "
        #              "--dev ${TARGETS[1]} "
        #              "--test ${TARGETS[2]} "
        #              "--random_seed ${RANDOM_SEED} "
        #              "--min_len ${MIN_LEN} "
        #              "--max_len ${MAX_LEN} "
        #              "--sample_method ${SAMPLE_METHOD} "
        #              "--samples_per_document ${SAMPLES_PER_DOCUMENT} "
        #              "--train_proportion ${TRAIN_PROPORTION} "
        #              "--dev_proportion ${DEV_PROPORTION} "
        #              "--test_proportion ${TEST_PROPORTION}"
        # 	)
        # ),
        "GenerateDocSplits" : Builder(
            action = ("python scripts/generate_doc_splits.py "
                      "--input ${SOURCES[0]} "
                      "--train ${TARGETS[0]} "
                      "--dev ${TARGETS[1]} "
                      "--test ${TARGETS[2]} "
                      "--random_seed ${RANDOM_SEED} "
                      "--min_len ${MIN_LEN} "
                      "--max_len ${MAX_LEN} "
                      "--sample_method ${SAMPLE_METHOD} "
                      "--samples_per_document ${SAMPLES_PER_DOCUMENT} "
                      "--train_proportion ${TRAIN_PROPORTION} "
                      "--dev_proportion ${DEV_PROPORTION} "
                      "--test_proportion ${TEST_PROPORTION} "
            ),
        ),
        #"SampleSequence": Builder(
        #    action="python scripts/sample_sequence.py --input ${SOURCES} --output ${TARGETS} --min_seq ${MIN_SEQ} --max_seq ${MAX_SEQ} --sample_method ${SAMPLE_METHOD} --samples ${SAMPLES} --seed ${FOLDS} "
        #),
        # "TrainClassifierModel": Builder(
        #     action="python scripts/train_model.py --train ${SOURCES[0]} --test ${SOURCES[1]} --model ${MODEL} --model_name ${SAVE_NAME} --emb_dim ${EMB_DIM} --num_epochs ${EPOCHS} --result ${TARGETS[0]} --errors ${TARGETS[1]} --batch ${BATCH} --confusion ${CM}",
        # ),
        # "TrainSequenceModel": Builder(
        #     action= (
        #         	"python scripts/train_sequence_model.py "
        #     		"--train ${SOURCES[0]} "
        #        		"--dev ${SOURCES[1]} "
        #          	"--test ${SOURCES[2]} "
        #           	"--output ${TARGETS[0]} "
        #             "--model ${ARCHITECTURE} "
        #             "--num_epochs ${EPOCHS} "
        #             "--batch_size ${BATCH_SIZE} "
        #             "--dropout ${DROPOUT} "
        #             "--granularity ${GRANULARITY} "
        #             "--boundary_type ${BOUNDARY_TYPE}"
        # 	)
        # ),
        "TrainHRNN": Builder(
            action= (
                        "python scripts/train_generic_hrnn.py "
                        "--train ${SOURCES[0]} "
                        "--dev ${SOURCES[1]} "
                        "--test ${SOURCES[2]} "
                        "--train_output ${TARGETS[0]} "
                        "--dev_output ${TARGETS[1]} "
                        "--training_summary ${TARGETS[2]} "
                        "--training_stats ${TARGETS[3]} "
                        "--model ${TARGETS[4]} "
                        "--teacher_ratio ${TEACHER_RATIO} "
                        "--hrnn_layer_names ${HRNN_LAYER_NAMES} "
                        "--threshold ${THRESHOLD} "
                        "--temperature ${TEMPERATURE} "
                        "--num_epochs ${EPOCHS} "
                        "--batch_size ${BATCH_SIZE} "
                        "--dropout ${DROPOUT} "
                        "--layer_weights ${LAYER_WEIGHTS} "
                        "${BALANCE_POS_NEG and '--balance_pos_neg $BALANCE_POS_NEG' or ''}"
                    )
        ),
        
        "ComputeROCMetrics": Builder(
			action=(
				"python scripts/compute_roc_metrics.py "
				"--input ${SOURCES[0]} "
    			"--threshold_metrics ${TARGETS[0]} "
				"--roc_by_layer ${TARGETS[1:]} "
				"--hrnn_layer_names ${HRNN_LAYER_NAMES}"
			)
		),
        
        "ComputeConfidenceMetrics": Builder(
			action=(
				"python scripts/compute_text_confidence_metrics.py "
				"--input ${SOURCES[0]} "
    			"--confidence_matrix ${TARGETS[0]} "
				"--hrnn_layer_names ${HRNN_LAYER_NAMES} "
    			"--threshold ${THRESHOLD}"
			)
		),
        
        "PlotTrainingMetrics": Builder(
			action=(
				"python scripts/plot_training_metrics.py "
				"--input ${SOURCES[0]} "
				"--loss_curves ${TARGETS[0:1]} "
				"--layer_loss_curves ${TARGETS[1:3]} "
				"--hrnn_layer_names ${HRNN_LAYER_NAMES}"
			)
		),
        
        "BuildVisualizations": Builder(
			action=(
				"python scripts/build_visualizations.py "
				"--input ${SOURCES[0]} "
				"--hrnn_visualizations ${TARGETS} "
				"--hrnn_layer_names ${HRNN_LAYER_NAMES} "
				"--threshold ${THRESHOLD}"
			)
		),
        
        "FormatChapterBreak": Builder(
            action = (
                        "python scripts/format_chapterbreak.py "
                        "--input ${SOURCES[0]} "
                        "--output ${TARGETS[0]} "
                        "--splits ${SPLITS} "
                        "--filters ${FILTERS}"
                     )
        ),
        
        "EncodeSequencedTexts": Builder(
            action= (
                    "python scripts/encode_chapterbreak_data.py "
                     "--input ${SOURCES[0]} "
                     "--output ${TARGETS[0]} "
                     "--model_id ${MODEL_ID} "
                     "--max_toks ${MAX_TOKS} "
                     "--batch_size ${BATCH_SIZE}"
            )
        ),
        
        "Evaluate": Builder(
            action = (
                "python scripts/evaluate.py "
                "--input ${SOURCES[0]} "
                "--model ${SOURCES[1]} "
                "--output ${TARGETS[0]} "
                "--batch_size ${BATCH_SIZE} "
                "--threshold ${THRESHOLD}"
            )
        )
        # "BuildVisualizations": Builder(
        #     action="python scripts/generate_report.py --input ${SOURCES[0]} --output ${TARGETS} --pg_path ${PG_PATH}"
        # ),
        # "GenerateReport": Builder(
        #     action="python scripts/generate_report.py --input ${SOURCES[0]} --output ${TARGETS} --pg_path ${PG_PATH}"
        # ),
        # "CollateResults": Builder(
        #     action="python scripts/collate_results.py --data ${SOURCES} --target ${TARGETS}"
        # ),
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

chicago_docs = env.ExtractStructureFromChicago(
    source = [],
    target = ["work/chicago_texts.jsonl.gz", "work/chicago_extracted_texts_catalog.txt.gz"],
    MIN_CHAPTERS = 3
)

transformed_chicago_docs = env.TransformExtractedData(
    source = [chicago_docs],
    target = ["work/transformed_chicago_texts.jsonl.gz"],
    FILTERS = env["SENTENCE_FILTERS"], # repr(json.dumps(env["SENTENCE_FILTERS"])),
    MIN_AVG_CH_LEN = 2,
    MERGE_PARAGRAPHS = True,
    **cpu_task_config("transform_docs", "12:00:00", "64G")
)

chapterbreak_data = env.FormatChapterBreak(
    source = [env.get("CHAPTERBREAK_FILE")],
    target = [f"work/evaluation/chapter_break_data.jsonl.gz"],
    SPLITS = ["pg19", "ao3"],
    FILTERS = repr(env["SENTENCE_FILTERS"])
)

for model in env.get("MODELS", []):
    enc_texts = env.EncodeData(
        source = [transformed_chicago_docs],
        target = ["work/${MODEL_NAME}/encoded_texts.jsonl.gz"],
        MODEL_NAME=model["name"],
        MODEL_ID=model["id"],
        BATCH_SIZE=model.get("BATCH_SIZE", 64),
        **gpu_task_config("encode_data", "12:00:00", "32G"),
    )
    
    encoded_chapterbreak_data = env.EncodeSequencedTexts(
        source = [chapterbreak_data],
        target = ["work/${MODEL_NAME}/encoded_chapterbreak.jsonl.gz"],
        MODEL_NAME=model["name"],
        MODEL_ID=model["id"],
        BATCH_SIZE=model.get("BATCH_SIZE", 64),
        **gpu_task_config("encode_data", "12:00:00", "32G"),
    )
    
    if env.get("TOY_RUN", 0):
        print(f"Toy run: {env['TOY_RUN']}")
        if env.get("TOY_FILE", None):
            enc_texts = env.TruncateData(
                source = [enc_texts],
                target = [env.get("TOY_FILE")],
                RETAIN_LINES = env["TOY_RUN"]
            )
        else:
            enc_texts = env.File(env["TOY_RUN"])

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
                    SAMPLES_PER_DOCUMENT = 1,
                    TRAIN_PROPORTION = 0.8,
                    DEV_PROPORTION = 0.1,
                    TEST_PROPORTION = 0.1
                )
                
                for layer_weight in env.get("LAYER_WEIGHTS", []):
                    layer_weight_path = f"{env['WORK_DIR']}/{model['name']}/trained_model_output/layer_weight_{layer_weight}"
                    visualization_path = f"{layer_weight_path}/visualizations"
                    roc_path = f"{layer_weight_path}/roc_metrics"
                    
                    train_guesses, dev_guesses, training_summary, training_metrics, trained_model = env.TrainHRNN(
                        source = [train, dev, test],
                        target = [f"{layer_weight_path}/guesses/train_guesses.pkl",
                                  f"{layer_weight_path}/guesses/dev_guesses.pkl",
                                  f"{layer_weight_path}/results/training_results.txt",
                                  f"{layer_weight_path}/results/training_metrics.pkl",
                                  f"{layer_weight_path}/model/model_state.pth"],
                        MODEL_NAME = model["name"],
                        BATCH_SIZE = 10,
                        DROPOUT = 0.6,
                        TEACHER_RATIO = 1.0,
                        EPOCHS = env.get("EPOCHS"),
                        THRESHOLD =  env["THRESHOLD"],
                        BALANCE_POS_NEG = [1.0, 1.0],
                        LAYER_WEIGHTS = [1.0, layer_weight],
                        **gpu_task_config("train_hrnn", "12:00:00", "32G"),
                    )
                    
                    training_metrics_visualizations = env.PlotTrainingMetrics(
						source = [training_metrics],
						target = [
							f"{layer_weight_path}/results/loss_curves.png",
       						f"{layer_weight_path}/results/par_layer_loss_curves.png",
             				f"{layer_weight_path}/results/chapter_layer_loss_curves.png",
						]
					)
                    
                    # Get ROC metrics and visualizations
                    roc_visualizations = [f"{roc_path}/roc_curve_for_{layer_name}" for layer_name in env["HRNN_LAYER_NAMES"]]
                    
                    roc_metrics = env.ComputeROCMetrics(
						source = [dev_guesses],
						target = [f"{roc_path}/optimal_thresholds.json",
                                  roc_visualizations],
					)
                    
                    # Get confidence matrix
                    confidence_matrix = env.ComputeConfidenceMetrics(
						source = [dev_guesses],
						target = [f"{layer_weight_path}/confidence_matrix.json"],
      					THRESHOLD = env["THRESHOLD"],
					)
                    
                    # Get boundary visualizations
                    layer_visualizations = [f"{visualization_path}/visualization_{num}" for num in range(env["VIS_NUM"])]
                    
                    boundary_visualizations = env.BuildVisualizations(
                        source = [dev_guesses],
                        target = [layer_visualizations],
                        THRESHOLD = env["THRESHOLD"],
                        NUM_LAYERS = len(env["HRNN_LAYER_NAMES"])
                    )
                    
                    chapterbreak_guesses = env.Evaluate(
                    	source = [encoded_chapterbreak_data, trained_model],
                    	target = [f"{layer_weight_path}/chapterbreak_guesses.pkl"],
                    	BATCH_SIZE = 10,
                    	THRESHOLD =  env["THRESHOLD"]
                    )
                    
                    
                    
                    

                    
                    # target = ["${WORK_DIR}/${MODEL_NAME}/model_guesses.pkl",
                    #             "${WORK_DIR}/${MODEL_NAME}/training_results.txt",
                    #             "${WORK_DIR}/${MODEL_NAME}/model_state.pth"],
                
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
                
