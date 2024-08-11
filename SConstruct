import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json

import jsonlines
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
    ("LOCAL_DATA", "", ""),
    ("PG_CATALOG", "", "pg_catalog.csv"),
    ("ENC_MODEL_NAME", "", "bert-base-uncased"),
    ("ENCODE_SIZE", "Maximum number of texts to encode", 3000),
    ("MAX_TOKS", "", 512),
    ("EMB_DIM", "Size of BERT embedding", 1536),
    ("CLUSTERS", "Number of clusters when grouping", 3),
    ("SPLIT_SIZE", "Maximum number of texts to use during train/test/split", 600),
    ("SPLIT_RATIOS", "Ratios to use for train/dev/test", [0.8, 0.1, 0.1]),
    ("SAMPLES", "Number of chapters to sample from text", 5),
    ("EXPERIMENT_TYPE", "", "sequence_tagger"),  # sequence_tagger, cross-classifier, classifier, contrastive
    ("SEQ_LENGTHS", "", [(30,40)]),
    ("MODEL_NAMES", "", []),
    ("EPOCHS", "Number of epochs to train model", 100),
    ("EXPERIMENTS", "", []),
    ("TITLE_FILTERS", "", [".*\\bpoe(m|t|s).*", ".*\\bballad.*", ".*\\bverse.*"]),
    ("REQUIRED_PATTERNS", "", [".*fiction.*"]),
    ("LOC_TAGS_TO_KEEP", "Library of Congress tags indicating a text may be kept", ["PS", "PE"]),
)

# Overrides
vars.AddVariables(
    ("PG_FILE", "", ""), 
    ("ENC_FILE", "", ""),
    ("SEQ_FILE", "", ""),
    ("TOY_FILE", "", ""),
    ("TOY_RUN", "", False)
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
        "ExtractTextsFromGutenberg" : Builder(
            action="python scripts/extract_texts_from_gutenberg.py --gutenberg_path ${GUTENBERG_PATH} --output ${TARGETS} --title_filters ${' '.join(['\"' + x + '\"' for x in TITLE_FILTERS])} --loc_tags_to_keep ${LOC_TAGS_TO_KEEP} --must_occur ${' '.join(['\"' + x + '\"' for x in REQUIRED_PATTERNS])}",
        ),
        "ExtractStructureFromTexts" : Builder(
            action="python scripts/extract_structure_from_texts.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        
        #"ProcessPGLocal" : Builder(
        #    action="python scripts/extract_from_gutenberg.py --base_dir ${LOCAL_TEST} --input ${SOURCES} --output ${TARGETS} --local $LOCAL",
        #),
        #"ExtractFromGutenberg" : Builder(
        #    action="python scripts/extract_from_gutenberg.py --base_dir ${PG_DATAPATH} --catalog ${SOURCES} --output ${TARGETS[0]} --output_catalog ${TARGETS[1]} --local ${LOCAL}",
        #),
        "ProcessWW" : Builder(
            action="python scripts/process_ww.py --output ${TARGETS} --data_path ${SOURCES}",
        ),
        "CreateSample": Builder (
            action="python scripts/create_sample.py --inputs ${SOURCES} --output ${TARGETS[0]} --shuffle ${SHUFFLE} --sample_size ${SAMPLE_SIZE} --seed ${FOLDS}"
        ),
        "EncodeData": Builder(
            action="python scripts/encode_data.py --input ${SOURCES[0]} --output ${TARGETS[0]} --model_name ${ENC_MODEL_NAME} --max_toks ${MAX_TOKS}",
        ),
        "ClusterCollection": Builder(
            action="python scripts/cluster_collection.py --collection ${SOURCES[0]} --pg_path ${PG_COLLECTION} --compressed_output ${TARGETS[0]} --sample_output ${TARGETS[1]} --image ${TARGETS[2]} --clusters ${CLUSTERS}"
        ),
        "GenerateSequence": Builder(
            action="python scripts/generate_sequence.py --input ${SOURCES} --output ${TARGETS[0]} --readable ${TARGETS[1]} --granularity ${GRANULARITY} --cluster ${CLUSTER} --seed ${FOLDS}"
        ),
        "SplitData": Builder(
            action="python scripts/split_data.py --inputs ${SOURCES} --outputs ${TARGETS} --sample_size ${SPLIT_SIZE} --split_ratio ${SPLIT_RATIOS} --cd ${CROSS_DOMAIN} --seed ${FOLDS}"
        ),
        "CreateClassifierDatapoints": Builder(
            action="python scripts/create_datapoints.py --input ${SOURCES} --output ${TARGETS} --samples ${SAMPLES} --same ${SAME_CH} --context_size ${CONTEXT} --seed ${FOLDS} --min_len ${MIN_LEN}"
        ),
        "SampleSequence": Builder(
            action="python scripts/sample_sequence.py --input ${SOURCES} --output ${TARGETS} --min_seq ${MIN_SEQ} --max_seq ${MAX_SEQ} --sample_method ${SAMPLE_METHOD} --samples ${SAMPLES} --seed ${FOLDS} "
        ),
        "TrainClassifierModel": Builder(
            action="python scripts/train_model.py --train ${SOURCES[0]} --test ${SOURCES[1]} --model ${MODEL} --model_name ${SAVE_NAME} --emb_dim ${EMB_DIM} --num_epochs ${EPOCHS} --result ${TARGETS[0]} --errors ${TARGETS[1]} --batch ${BATCH} --confusion ${CM}",
        ),
        "TrainSequenceModel": Builder(
            action="python scripts/train_sequence_model.py --data ${SOURCES} --model_save_name ${SAVE_NAME} --visualizations ${VISUALIZATIONS} --output_data ${TARGETS} --model ${MODEL}  --emb_dim ${EMB_DIM} --output_layers ${OUTPUT_LAYERS} --classes ${CLASSES} --num_epochs ${EPOCHS} --batch ${BATCH}",
        ),
        "GenerateReport": Builder(
            action="python scripts/generate_report.py --input ${SOURCES[0]} --output ${TARGETS} --pg_path ${PG_PATH}"
        ),
        "CollateResults": Builder(
            action="python scripts/collate_results.py --data ${SOURCES} --target ${TARGETS}"
        ),
        "SanityCheck": Builder(
			action="python scripts/sanity_check.py --splits ${SOURCES} --output ${TARGETS}"
		)
    }
)

texts = env.ExtractTextsFromGutenberg(
    "work/texts.jsonl.gz",
    [],
)

structure = env.ExtractStructureFromTexts(
    "work/structure.jsonl.gz",
    texts
)

# sys.exit()

# # Extract documents from Project Gutenberg
# if env.get("LOCAL_DATA", None):
#     print("Debugging run")
#     pg_local = env.ProcessPGLocal(source = env["PG_CATALOG"] ,
#                                   target = ["work/gutenberg.jsonl", "work/test.txt"])
# else:
#     if env.get("PG_FILE", None):
#         extracted_docs = env.File(env["PG_FILE"])
#     else:
#         extracted_docs, extracted_catalog = env.ExtractFromGutenberg(source = env["PG_CATALOG"] ,
#                                                                      target = ["work/data/gutenberg_file.jsonl", "work/data/gutenberg_catalog_extracted.txt"],
#                                                                      STEAMROLLER_QUEUE=env["CPU_QUEUE"],
#                                                                      STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
#                                                                      STEAMROLLER_MEMORY="32G")
#         # Downsample Step
#         pg_sample = env.CreateSample(source = [extracted_docs],
#                                      target = ["work/data/gutenberg_sample.jsonl"],
#                                      SAMPLE_SIZE = 100000,
#                                      SHUFFLE = 1,
#                                      STEAMROLLER_QUEUE=env["CPU_QUEUE"],
#                                      STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
#                                      STEAMROLLER_MEMORY="32G")

# # Encode documents and create data lake
# if env.get("ENC_FILE", None):
#     pg_enc = env.File(env["ENC_FILE"])
# else:
#     pg_enc = env.EncodeData(source = [pg_sample],
#                             target = ["work/data/gutenberg_encoded.jsonl.gz"],
#                             STEAMROLLER_GPU_COUNT=1,
#                             STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
#                             STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
#                             STEAMROLLER_TIME = "12:00:00",
#                             STEAMROLLER_MEMORY="32G")

# # Switch to toy file of encoded data if toy file exists
# if env["TOY_RUN"] and env.get("TOY_FILE"):
#     print("Switched to toy file")
#     pg_enc = env.File(env["TOY_FILE"])
# elif env["TOY_RUN"]:
#     pg_enc = env.CreateSample(source = [pg_enc],
#                               target = ["work/data/toy_1600.jsonl.gz"],
#                               SHUFFLE = 0,
#                               SAMPLE_SIZE = 1600,
#                               STEAMROLLER_QUEUE=env["CPU_QUEUE"],
# 							  STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
# 							  STEAMROLLER_MEMORY="32G")

# # # Create splits of the data (train, dev, test)
# exp_type = env["EXPERIMENT_TYPE"]

# train_paths, dev_paths, test_paths = [], [], []

# for i in range(env["FOLDS"]):
#     train_paths.append(f"work/experiments/{exp_type}/trial_{i}/shuffle_train.jsonl.gz")
#     dev_paths.append(f"work/experiments/{exp_type}/trial_{i}/shuffle_dev.jsonl.gz")
#     test_paths.append(f"work/experiments/{exp_type}/trial_{i}/shuffle_test.jsonl.gz")

# output_paths = train_paths + dev_paths + test_paths

# # # Cluster and filter
# # compressed_collection, sample_clusters, cluster_graph = env.ClusterCollection(source = [pg_enc],
# #                                                                               target = [f"work/data/encoded_clustered_collection.jsonl.gz", f"work/data/sample_clusters.json", f"work/data/cluster_graph.png"],
# #                                                                               PG_COLLECTION = extracted_docs)


# # # Flatten data to sequence representation
# seq_file = env.GenerateSequence(source = [pg_enc],
#                                 target = [f"work/data/flattened_encodings.jsonl.gz", f"work/data/readable-sequence.jsonl"],
#                                 GRANULARITY = 0,
#                                 CLUSTER = None,
#                                 STEAMROLLER_QUEUE=env["CPU_QUEUE"],
# 								STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
# 								STEAMROLLER_MEMORY="32G")

# experiment_results = []
# for experiment in env["EXPERIMENTS"]:
#     experiment_name, tagging_method, class_labels = experiment

#     train_test_files = env.SplitData(source = [seq_file], 
#                                      target = [output_paths])
    
#     for n in range(env["FOLDS"]):
#         for min_seq, max_seq in env["SEQ_LENGTHS"]:
#             train_set = env.SampleSequence(source = train_test_files[n],
#                                            target = f"work/experiments/{exp_type}/{tagging_method}/trial_{n}/{experiment_name}/data/{min_seq}-{max_seq}/train.jsonl",
#                                            SAMPLES = 10,
#                                            MIN_SEQ = min_seq,
#                                            MAX_SEQ = max_seq,
#                                            SAMPLE_METHOD = "random_subseq",
#                                            STEAMROLLER_QUEUE=env["CPU_QUEUE"],
# 										   STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
# 										   STEAMROLLER_MEMORY="32G")
#             dev_set = env.SampleSequence(source = train_test_files[env["FOLDS"]+n],
#                                         target = f"work/experiments/{exp_type}/{tagging_method}/trial_{n}/{experiment_name}/data/{min_seq}-{max_seq}/dev.jsonl",
#                                         SAMPLES = 5,
#                                         MIN_SEQ = min_seq,
#                                         MAX_SEQ = max_seq,
#                                         SAMPLE_METHOD = "random_subseq",
#                                         STEAMROLLER_QUEUE=env["CPU_QUEUE"],
# 										STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
# 										STEAMROLLER_MEMORY="32G")
#             # Create full-text samples
#             test_set = env.SampleSequence(source = train_test_files[env["FOLDS"]*2+n],
#                                           target = f"work/experiments/{exp_type}/{tagging_method}/trial_{n}/{experiment_name}/data/{min_seq}-{max_seq}/test.jsonl",
#                                           SAMPLES = 5,
#                                           MIN_SEQ = min_seq,
#                                           MAX_SEQ = max_seq,
#                                           SAMPLE_METHOD = "random_subseq",
#                                           STEAMROLLER_QUEUE=env["CPU_QUEUE"],
# 										  STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
# 										  STEAMROLLER_MEMORY="32G")
            
#             sanity_check_result = env.SanityCheck(source = [train_set, dev_set, test_set],
#                                                   target = [f"work/experiments/{exp_type}/{tagging_method}/trial_{n}/{experiment_name}/data/{min_seq}-{max_seq}/sanity_check_output.txt"])
            
#             for model_name in env["MODEL_NAMES"]:
#                 result = env.TrainSequenceModel(source = [train_set, dev_set, test_set],
# 												target = f"work/experiments/{exp_type}/{tagging_method}/trial_{n}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/results.pickle", 
# 												SAVE_NAME=f"work/experiments/{exp_type}/{tagging_method}/trial_{n}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/best_model/{min_seq}-{max_seq}-model.pt",
# 												VISUALIZATIONS=f"work/experiments/{exp_type}/{tagging_method}/trial_{n}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/visualizations",
# 												MODEL = model_name,
# 												EMB_DIM = 768,
# 												BATCH=32,
# 												CLASSES = class_labels,
# 												OUTPUT_LAYERS = 2,
#             									STEAMROLLER_QUEUE=env["CPU_QUEUE"],
# 												STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"],
# 												STEAMROLLER_MEMORY="32G")
#                 experiment_results.append(result)
#                 # report = env.GenerateReport(source = [errors],
#                 #                             target = f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/trial_{n}/result/{min_seq}-{max_seq}-errors.txt",
#                 #                             MODEL_TYPE = model_name,
#                 #                             PG_PATH = env["PG_FILE"])
                
# # report = env.CollateResults(source = experiment_results,
# #                             target = f"work/experiments/{exp_type}/final_report")















# # results = []
# # if "classifier" in exp_type:
# #     if "cross" in exp_type:
# #         print(f"Cross domain data shuffle") 
# #         train_test_files = env.ShuffleAndSplitData(source = [pg_enc], 
# #                                             target = [output_paths], 
# #                                             CROSS_DOMAIN = ww_enc)
# #     else:
# #         print(f"Within domain data shuffle")
# #         print(f"PG ENC HERE: {pg_enc}")
# #         train_test_files = env.ShuffleAndSplitData(source = [pg_enc], 
# #                                                    target = [output_paths], 
# #                                                    CROSS_DOMAIN = "")
                                            
    
# #     for n in range(env["FOLDS"]):
# #         for context_size in env["CONTEXT_WINDOW_SIZE"]:
# #             print(f"Context size: {context_size}")
# #             train_set = env.CreateClassifierDatapoints(source = train_test_files[n], 
# #                                             target = f"work/experiments/{exp_type}/trial_{n}/train-{context_size}.jsonl", 
# #                                             CONTEXT=context_size,
# #                                             MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
# #                                             GRID_MEMORY="4G")
# #             test_set = env.CreateClassifierDatapoints(source = train_test_files[env["FOLDS"]+n], 
# #                                             target = f"work/experiments/{exp_type}/trial_{n}/test-{context_size}.jsonl", 
# #                                             CONTEXT=context_size,
# #                                             MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
# #                                             GRID_MEMORY="4G")
# #             result, errors = env.TrainModel(source = [train_set, test_set], 
# #                                     target = [f"work/experiments/{exp_type}/trial_{n}/result/{context_size}.txt",
# #                                                 f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-error-ids.txt"], 
# #                                     SAVE_NAME=f"work/experiments/{exp_type}/trial_{n}/best_model/{context_size}.pt",
# #                                     CM = f"work/experiments/{exp_type}/confusion-{context_size}.png",
# #                                     MODEL="contrastive",
# #                                     GRID_GPU_COUNT=1,
# #                                     GRID_ACCOUNT="tlippin1_gpu",
# #                                     GRID_QUEUE="a100",
# #                                     GRID_TIME="45:00")
# #             report = env.GenerateReport(source = [errors],
# #                                         target = f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-errors.txt",
# #                                         PG_PATH = env["PG_FILE"])
# #             results.append(result)

# # elif "sequence_tagger" in exp_type:
# #     compressed_collection, sample_clusters, cluster_graph = env.ClusterCollection(source = [pg_enc],
# #                                                                                   target = [f"work/data/encoded_clustered_collection.gz", f"work/data/sample_clusters.json", f"work/data/cluster_graph.png"],
# #                                                                                   PG_COLLECTION = env["PG_FILE"])
    
# #     seq_file = env.GenerateSequence(source = compressed_collection,
# #                                     target = [f"work/data/generated_sequence.json.gz", f"work/data/readable-sequence.json"],
# #                                     GRANULARITY = 0,
# #                                     CLUSTER = None)
    
# #     experiment_results = []
# #     for experiment in env["EXPERIMENTS"]:
# #         tagging_method, granularity, class_labels = experiment
        
        
# #         output_train_paths, output_dev_paths, output_test_paths = [], [], []

# #         for i in range(env["FOLDS"]):
# #             output_train_paths.append(f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/data/trial_{i}/shuffle_train.jsonl.gz")
# #             output_dev_paths.append(f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/data/trial_{i}/shuffle_dev.jsonl.gz")
# #             output_test_paths.append(f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/data/trial_{i}/shuffle_test.jsonl.gz")
        
# #         output_train_paths.extend(output_dev_paths)
# #         output_train_paths.extend(output_test_paths)
# #         output_paths = output_train_paths
        
# #         # print(f"OUTPUT PATHS: {output_paths}")
# #         print(f"Sequence shuffle")
# #         train_test_files = env.ShuffleAndSplitData(source = [seq_file], 
# #                                                    target = [output_paths])
        

# #         print(f"Train test files: {train_test_files}")
# #         for n in range(env["FOLDS"]):
# #             for min_seq, max_seq in env["SEQ_LENGTHS"]:
# #                 train_set = env.SampleSequence(source = train_test_files[n],
# #                                                 target = f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/data/trial_{n}/{min_seq}-{max_seq}/train.jsonl",
# #                                                 SAMPLES = 20,
# #                                                 MIN_SEQ = min_seq,
# #                                                 MAX_SEQ = max_seq,
# #                                                 SAMPLE_METHOD = "random_subseq")
# #                 dev_set = env.SampleSequence(source = train_test_files[env["FOLDS"]+n],
# #                                             target = f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/data/trial_{n}/{min_seq}-{max_seq}/dev.jsonl",
# #                                             SAMPLES = 5,
# #                                             MIN_SEQ = min_seq,
# #                                             MAX_SEQ = max_seq,
# #                                             SAMPLE_METHOD = "random_subseq")
# #                 # Create full-text samples
# #                 test_set = env.SampleSequence(source = train_test_files[env["FOLDS"]*2+n],
# #                                                 target = f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/data/trial_{n}/{min_seq}-{max_seq}/test.jsonl",
# #                                                 SAMPLES = 5,
# #                                                 MIN_SEQ = min_seq,
# #                                                 MAX_SEQ = max_seq,
# #                                                 SAMPLE_METHOD = "random_subseq")
                
# #                 for model_name in env["MODEL_NAMES"]:
# #                     result = env.TrainSequenceModel(source = [train_set, dev_set, test_set],
# #                                                             target = f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/trial_{n}/results.pickle", 
# #                                                             SAVE_NAME=f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/trial_{n}/best_model/{min_seq}-{max_seq}-model.pt",
# #                                                             VISUALIZATIONS=f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/trial_{n}/visualizations",
# #                                                             MODEL = model_name,
# #                                                             EMB_DIM = 768,
# #                                                             BATCH=32,
# #                                                             CLASSES = class_labels,
# #                                                             OUTPUT_LAYERS = 2)
# #         #             experiment_results.append(result)
# #                     # report = env.GenerateReport(source = [errors],
# #                     #                             target = f"work/experiments/{exp_type}/{tagging_method}/{experiment_name}/{min_seq}-{max_seq}/{model_name}/trial_{n}/result/{min_seq}-{max_seq}-errors.txt",
# #                     #                             MODEL_TYPE = model_name,
# #                     #                             PG_PATH = env["PG_FILE"])
                    
# #     # report = env.CollateResults(source = experiment_results,
# #     #                             target = f"work/experiments/{exp_type}/final_report")
# # elif "contrastive" in exp_type:
# #     print(f"Within domain data shuffle")
# #     print(f"PG ENC HERE: {pg_enc}")
# #     train_test_files = env.ShuffleAndSplitData(source = [pg_enc], 
# #                                         target = [output_paths], 
# #                                         CROSS_DOMAIN = "")
    
# # else:
# #     print("Incorrect experiment type")

    

# # results = []

# # seq_file = env.GenerateSequence(source = compressed_collection,
#         #                                 target = [f"work/data/{experiment_name}.json.gz", f"work/data/readable-{experiment_name}.json"],
#         #                                 GRANULARITY = granularity,
#         #                                 CLUSTER = 3)


# # for n in range(env["FOLDS"]):
# #     if env["SEQ"]:
# #         print(f"Train test files: {train_test_files}")
# #         for min_seq, max_seq in env["SEQ_LENGTHS"]:
# #             boundary = "boundary_context" if env["BOUNDARY_CONTEXT"] else "no_boundary_context"
# #             train_set = env.CreateSequenceData(source = train_test_files[n],
# #                                             target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{min_seq}-{max_seq}/train.jsonl",
# #                                             MIN_SEQ = min_seq,
# #                                             MAX_SEQ = max_seq)
# #             test_set = env.CreateSequenceData(source = train_test_files[env["FOLDS"]+n],
# #                                             target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{min_seq}-{max_seq}/test.jsonl",
# #                                             MIN_SEQ = min_seq,
# #                                             MAX_SEQ = max_seq)
# #             for model_name in env["MODEL_NAMES"]:
# #                 result, errors = env.TrainModel(source = [train_set, test_set],
# #                                         target = [f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-result.txt",
# #                                                     f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-error-ids.txt"], 
# #                                         SAVE_NAME=f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/best_model/{min_seq}-{max_seq}-model.pt",
# #                                         MODEL = model_name,
# #                                         CM = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/confusion_matrix-{min_seq}-{max_seq}.png",
# #                                         EMB_DIM = 768,
# #                                         BATCH=32,
# #                                         GRID_GPU_COUNT=1,
# #                                         GRID_ACCOUNT="tlippin1_gpu",
# #                                         GRID_QUEUE="a100")
# #                 report = env.GenerateReport(source = [errors],
# #                                             target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-errors.txt",
# #                                             MODEL_TYPE = model_name,
# #                                             PG_PATH = env["PG_FILE"])
# #     else:
# #         for context_size in env["CONTEXT_WINDOW_SIZE"]:
# #             print(f"Context size: {context_size}")
# #             train_set = env.CreateDatapoints(source = train_test_files[n], 
# #                                             target = f"work/experiments/{exp_type}/trial_{n}/train-{context_size}.jsonl", 
# #                                             CONTEXT=context_size,
# #                                             MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
# #                                             GRID_MEMORY="4G")
# #             test_set = env.CreateDatapoints(source = train_test_files[env["FOLDS"]+n], 
# #                                             target = f"work/experiments/{exp_type}/trial_{n}/test-{context_size}.jsonl", 
# #                                             CONTEXT=context_size,
# #                                             MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
# #                                             GRID_MEMORY="4G")
# #             result, errors = env.TrainModel(source = [train_set, test_set], 
# #                                     target = [f"work/experiments/{exp_type}/trial_{n}/result/{context_size}.txt",
# #                                                 f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-error-ids.txt"], 
# #                                     SAVE_NAME=f"work/experiments/{exp_type}/trial_{n}/best_model/{context_size}.pt",
# #                                     CM = f"work/experiments/{exp_type}/confusion-{context_size}.png",
# #                                     MODEL="contrastive",
# #                                     GRID_GPU_COUNT=1,
# #                                     GRID_ACCOUNT="tlippin1_gpu",
# #                                     GRID_QUEUE="a100",
# #                                     GRID_TIME="45:00")
# #             report = env.GenerateReport(source = [errors],
# #                                         target = f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-errors.txt",
# #                                         PG_PATH = env["PG_FILE"])
# #             results.append(result)

