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
from steamroller import Environment
import jsonlines


vars = Variables("custom.py")
vars.AddVariables(
    ("WW_DATAPATH", "", "/export/data/english/women_writers.tgz"), # correct
    ("PG_DATAPATH", "", "~/corpora/gutenberg/"),
    ("LOCAL_DATA", "", ["./data/warren.adulateur.xml", "./data/haywood.eovaai.xml", "./data/smith.manor.xml"]),
    ("PG_CATALOG", "", "pg_catalog.csv"),
    ("ENC_MODEL_NAME", "", "bert-base-uncased"),
    ("MAX_TOKS", "", 512),
    ("LOCAL", "", "False"),
    ("DATA_SIZE", "Number of texts to encode", 3000),
    ("SAMPLE_SIZE", "Number of texts to grab chapters from", 600),
    ("TRAIN_TEST_SPLIT", "", 0.8),
    ("SAMPLES", "Number of chapters to sample from text", 5),
    ("EMB_DIM", "Size of BERT embedding", 1536),
    ("EPOCHS", "", 50),
    ("SAME_CH", "", "True"),
    ("CONTEXT_WINDOW_SIZE", "", [2, 10, 30, 50]), # [2, 10, 50]
    ("CD", "", ""), # cd or no_cd
    ("NUM_TRIALS", "Number of replications", 1),
    ("SAMPLE_SEED", "Going from gutenberg to encoding", 1),
    ("DPS_SEED", "Going from encoding to datapoints", 1),
    ("SHUFFLE_SEED", "Shuffle", 1),
    ("PG_FILE", "", "./work/data/gutenberg-lines.jsonl"), #  "./work/data/pg_sample.jsonl"
    ("ENC_FILE", "", "./work/data/newest_encoded.jsonl.gz"), # ./work/data/gutenberg_encoded.jsonl.gz # "work/data/updated_gutenberg_encoded.jsonl.gz"
    ("SEQ_FILE", "", "./work/data/seq_encoded_unordered.json.gz"), # ./work/data/seq_encoded_ordered_correct.json.gz # , ./work/data/seq_encoded_ordered_correct.json.gz" # "./work/data/seq_encoded_correct.json.gz"
    ("NUM_ENC", "", 2),
    ("EXPERIMENT_TYPE", "", "sequence_tagger"),  # sequence_tagger, cross-classifier, classifier, contrastive
    ("SEQ_LENGTHS", "", [(20,30)]),  # , (30,40), (50,60)]),
    ("BOUNDARY_CONTEXT", "", 1),
    ("MODEL_NAMES", "", ["sequence_tagger", "sequence_tagger_with_bahdanau_attention"]),
    ("USE_GRID", "", 0),
    ("GRID_TYPE", "", "slurm"),
    ("GRID_GPU_COUNT", "", 0),
    ("GRID_MEMORY", "", "64G"),
    ("GRID_LABEL", "", "boundary_detection"),
    ("GRID_TIME", "", "48:00:00")
)


env = Environment(
    variables=vars,
    ENV=os.environ,
    BUILDERS={
        "ProcessPGLocal" : Builder(
            # action="python scripts/create_data.py --data_path ${SOURCES} --output ${TARGETS} --granularity $SEGMENT_BY_PG",
            action="python scripts/process_pg.py --base_dir ${LOCAL_TEST} --input ${SOURCES} --output ${TARGETS} --local $LOCAL",
        ),
        "ProcessPG" : Builder(
            # action="python scripts/create_data.py --data_path ${SOURCES} --output ${TARGETS} --granularity $SEGMENT_BY_PG",
            action="python scripts/process_pg.py --base_dir ${PG_DATAPATH} --catalog ${SOURCES} --output ${TARGETS} --local ${LOCAL}",
        ),
        "ProcessWW" : Builder(
            action="python scripts/process_ww.py --output ${TARGETS} --data_path ${SOURCES}",
        ),
        "CreateSample": Builder (
            action="python scripts/create_sample.py --inputs ${SOURCES} --output ${TARGETS[0]} --data_size ${DATA_SIZE} --seed ${SAMPLE_SEED}"
        ),
        "EncodeData": Builder(
            action="python scripts/encode_data.py --input ${SOURCES[0]} --output ${TARGETS[0]} --model_name ${ENC_MODEL_NAME} --max_toks ${MAX_TOKS}",
            GRID_GPU_COUNT=1,
            GRID_ACCOUNT="sli159_gpu",
            GRID_QUEUE="a100"
        ),
        "GenerateSequence": Builder(
            action="python scripts/generate_sequence.py --input ${SOURCES} --output ${TARGETS} --paragraph ${PARAGRAPH} --BOUNDARY_CONTEXT ${BOUNDARY_CONTEXT} --seed ${SEED}"
        ),
        "ShuffleData": Builder(
            action="python scripts/shuffle_data.py --inputs ${SOURCES} --outputs ${TARGETS} --sample_size ${SAMPLE_SIZE} --split_ratio ${TRAIN_TEST_SPLIT} --cd ${CROSS_DOMAIN} --seed ${SHUFFLE_SEED}"
        ),
        "CreateClassifierDatapoints": Builder(
            action="python scripts/create_datapoints.py --input ${SOURCES} --output ${TARGETS} --samples ${SAMPLES} --same ${SAME_CH} --context_size ${CONTEXT} --seed ${DPS_SEED} --min_len ${MIN_LEN}"
        ),
        "CreateSequenceDatapoints": Builder(
            action="python scripts/create_sequence_data.py --input ${SOURCES} --output ${TARGETS} --samples ${SAMPLES} --seed ${DPS_SEED} --max_seq ${MAX_SEQ} --min_seq ${MIN_SEQ} --sample_method ${SAMPLE_METHOD} --upsampled_label ${UPSAMPLED_LABEL} --upsample_ratio ${UPSAMPLED_RATIO}"
        ),
        "TrainClassifierModel": Builder(
            action="python scripts/train_model.py --train ${SOURCES[0]} --test ${SOURCES[1]} --model ${MODEL} --model_name ${SAVE_NAME} --emb_dim ${EMB_DIM} --num_epochs ${EPOCHS} --result ${TARGETS[0]} --errors ${TARGETS[1]} --batch ${BATCH} --confusion ${CM}",
            GRID_GPU_COUNT=1,
            GRID_ACCOUNT="sli159_gpu",
            GRID_QUEUE="a100"
        ),
        "TrainSequenceModel": Builder(
            action="python scripts/train_sequence_model.py --train ${SOURCES[0]} --test ${SOURCES[1]} --model ${MODEL} --model_name ${SAVE_NAME} --emb_dim ${EMB_DIM} --num_epochs ${EPOCHS} --result ${TARGETS[0]} --errors ${TARGETS[1]} --batch ${BATCH} --confusion ${CM}",
            GRID_GPU_COUNT=1,
            GRID_ACCOUNT="sli159_gpu",
            GRID_QUEUE="a100"
        ),
        "GenerateReport": Builder(
            action="python scripts/generate_report.py --input ${SOURCES[0]} --output ${TARGETS} --pg_path ${PG_PATH}"
        )
    }
)

# Get data
if env.get("LOCAL") == "True":
    print("Debugging run")
    pg_local = env.ProcessPGLocal(source = env["PG_CATALOG"] ,
                                  target = ["work/gutenberg.jsonl", "work/test.txt"])
else:
    if env.get("ENC_FILE", None):
        pass
    elif env.get("PG_FILE", None):
        pg_sample = env.File(env["PG_FILE"])
    else:
        print("Gathering all data")
        pg_data = env.ProcessPG(source = env["PG_CATALOG"] ,
                                target = "work/data/gutenberg.jsonl")
        # ww_data = env.ProcessWW(source = env["WW_DATAPATH"],
        #                         target = ["work/data/womenwriters.jsonl"])
        # Downsample Step
        pg_sample = env.CreateSample(source = [pg_data],
                                    target = ["work/data/pg_sample.jsonl"])
        # ww_sample = env.CreateSample(source = [pg_data],
        #                              target = ["work/data/ww_sample.jsonl"])

# Encode Step
if env.get("ENC_FILE", None):
    pg_enc = env.File(env["ENC_FILE"])
    print("FOUND ENCODED FILE")
else:
    pg_enc = env.EncodeData(source = [pg_sample],
                            target = ["work/data/pg_encoded.json.gz"],
                            GRID_GPU_COUNT=1,
                            GRID_ACCOUNT="tlippin1_gpu",
                            GRID_QUEUE="a100",
                            GRID_TIME="12:00:00")
    # ww_enc = env.EncodeData(source = ww_sample,
    #                         target = f"work/data/ww_encoded.jsonl",
    #                         GRID_GPU_COUNT=1,
    #                         GRID_ACCOUNT="tlippin1_gpu",
    #                         GRID_QUEUE="a100")

exp_type = env["EXPERIMENT_TYPE"]
output_train_paths = []
output_test_paths = []

for i in range(env["NUM_TRIALS"]):
    output_train_paths.append(f"work/experiments/{exp_type}/trial_{i}/shuffle_train.jsonl.gz")
    output_test_paths.append(f"work/experiments/{exp_type}/trial_{i}/shuffle_test.jsonl.gz")

print(f"Train paths: {len(output_train_paths)}")
print(f"Test paths: {len(output_test_paths)}")

output_train_paths.extend(output_test_paths)
output_paths = output_train_paths
print(f"Len of output paths: {len(output_paths)}")

results = []
if "classifier" in exp_type:
    if "cross" in exp_type:
        print(f"Cross domain data shuffle") 
        train_test_files = env.ShuffleData(source = [pg_enc], 
                                            target = [output_paths], 
                                            CROSS_DOMAIN = ww_enc,
                                            SHUFFLE_SEED = env["SHUFFLE_SEED"])
    else:
        print(f"Within domain data shuffle")
        print(f"PG ENC HERE: {pg_enc}")
        train_test_files = env.ShuffleData(source = [pg_enc], 
                                            target = [output_paths], 
                                            CROSS_DOMAIN = "",
                                            SHUFFLE_SEED = env["SHUFFLE_SEED"])
                                            
    

    for context_size in env["CONTEXT_WINDOW_SIZE"]:
        print(f"Context size: {context_size}")
        train_set = env.CreateClassifierDatapoints(source = train_test_files[n], 
                                        target = f"work/experiments/{exp_type}/trial_{n}/train-{context_size}.jsonl", 
                                        CONTEXT=context_size,
                                        MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
                                        GRID_MEMORY="4G")
        test_set = env.CreateClassifierDatapoints(source = train_test_files[env["NUM_TRIALS"]+n], 
                                        target = f"work/experiments/{exp_type}/trial_{n}/test-{context_size}.jsonl", 
                                        CONTEXT=context_size,
                                        MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
                                        GRID_MEMORY="4G")
        result, errors = env.TrainModel(source = [train_set, test_set], 
                                target = [f"work/experiments/{exp_type}/trial_{n}/result/{context_size}.txt",
                                            f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-error-ids.txt"], 
                                SAVE_NAME=f"work/experiments/{exp_type}/trial_{n}/best_model/{context_size}.pt",
                                CM = f"work/experiments/{exp_type}/confusion-{context_size}.png",
                                MODEL="contrastive",
                                GRID_GPU_COUNT=1,
                                GRID_ACCOUNT="tlippin1_gpu",
                                GRID_QUEUE="a100",
                                GRID_TIME="45:00")
        report = env.GenerateReport(source = [errors],
                                    target = f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-errors.txt",
                                    PG_PATH = env["PG_FILE"])
        results.append(result)

elif "sequence_tagger" in exp_type:
    print(f"Sequence Tagger")
    if env.get("SEQ_FILE", None):
        seq_file = env.File(env["SEQ_FILE"])
    else:
        print(f"Generate sequence")
        seq_file = env.GenerateSequence(source = pg_enc,
                                        target = f"work/data/seq_encoded.jzon.gz",
                                        BOUNDARY_CONTEXT = 0,
                                        PARAGRAPH = 1,
                                        SEED = 0)
    
    print(f"Sequence shuffle")
    train_test_files = env.ShuffleData(source = [seq_file], 
                                        target = [output_paths], 
                                        CROSS_DOMAIN = "",
                                        SHUFFLE_SEED = env["SHUFFLE_SEED"])
    

    print(f"Train test files: {train_test_files}")
    for n in range(env["NUM_TRIALS"]):
        for min_seq, max_seq in env["SEQ_LENGTHS"]:
            boundary = "boundary_context" if env["BOUNDARY_CONTEXT"] else "no_boundary_context"
            train_set = env.CreateSequenceDatapoints(source = train_test_files[n],
                                            target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{min_seq}-{max_seq}/train.jsonl",
                                            MIN_SEQ = min_seq,
                                            MAX_SEQ = max_seq,
                                            UPSAMPLED_LABEL = [1],
                                        	UPSAMPLED_RATIO = 0.0,
                                            SAMPLE_METHOD = "from_beginning")
            # Create full-text samples
            test_set = env.CreateSequenceDatapoints(source = train_test_files[env["NUM_TRIALS"]+n],
                                            target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{min_seq}-{max_seq}/test.jsonl",
                                            SAMPLES = 1,
                                            MIN_SEQ = 1,
                                            MAX_SEQ = 1000,
                                            UPSAMPLED_LABEL = [1],
                                        	UPSAMPLED_RATIO = 0.0,
                                            SAMPLE_METHOD = "from_beginning")
            for model_name in env["MODEL_NAMES"]:
                result, errors = env.TrainSequenceModel(source = [train_set, test_set],
                                                        target = [f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-result.txt",
                                                                    f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-error-ids.txt"], 
                                                        SAVE_NAME=f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/best_model/{min_seq}-{max_seq}-model.pt",
                                                        MODEL = model_name,
                                                        CM = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/confusion_matrix-{min_seq}-{max_seq}.png",
                                                        EMB_DIM = 768,
                                                        BATCH=32,
                                                        GRID_GPU_COUNT=1,
                                                        GRID_ACCOUNT="tlippin1_gpu",
                                                        GRID_QUEUE="a100")
                report = env.GenerateReport(source = [errors],
                                            target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-errors.txt",
                                            MODEL_TYPE = model_name,
                                            PG_PATH = env["PG_FILE"])
elif "contrastive" in exp_type:
    print(f"Within domain data shuffle")
    print(f"PG ENC HERE: {pg_enc}")
    train_test_files = env.ShuffleData(source = [pg_enc], 
                                        target = [output_paths], 
                                        CROSS_DOMAIN = "",
                                        SHUFFLE_SEED = env["SHUFFLE_SEED"])
    
else:
    print("Incorrect experiment type")

    

# results = []


# for n in range(env["NUM_TRIALS"]):
#     if env["SEQ"]:
#         print(f"Train test files: {train_test_files}")
#         for min_seq, max_seq in env["SEQ_LENGTHS"]:
#             boundary = "boundary_context" if env["BOUNDARY_CONTEXT"] else "no_boundary_context"
#             train_set = env.CreateSequenceData(source = train_test_files[n],
#                                             target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{min_seq}-{max_seq}/train.jsonl",
#                                             MIN_SEQ = min_seq,
#                                             MAX_SEQ = max_seq)
#             test_set = env.CreateSequenceData(source = train_test_files[env["NUM_TRIALS"]+n],
#                                             target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{min_seq}-{max_seq}/test.jsonl",
#                                             MIN_SEQ = min_seq,
#                                             MAX_SEQ = max_seq)
#             for model_name in env["MODEL_NAMES"]:
#                 result, errors = env.TrainModel(source = [train_set, test_set],
#                                         target = [f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-result.txt",
#                                                     f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-error-ids.txt"], 
#                                         SAVE_NAME=f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/best_model/{min_seq}-{max_seq}-model.pt",
#                                         MODEL = model_name,
#                                         CM = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/confusion_matrix-{min_seq}-{max_seq}.png",
#                                         EMB_DIM = 768,
#                                         BATCH=32,
#                                         GRID_GPU_COUNT=1,
#                                         GRID_ACCOUNT="tlippin1_gpu",
#                                         GRID_QUEUE="a100")
#                 report = env.GenerateReport(source = [errors],
#                                             target = f"work/experiments/{exp_type}/trial_{n}/{boundary}/{model_name}/{min_seq}-{max_seq}/result/{min_seq}-{max_seq}-errors.txt",
#                                             MODEL_TYPE = model_name,
#                                             PG_PATH = env["PG_FILE"])
#     else:
#         for context_size in env["CONTEXT_WINDOW_SIZE"]:
#             print(f"Context size: {context_size}")
#             train_set = env.CreateDatapoints(source = train_test_files[n], 
#                                             target = f"work/experiments/{exp_type}/trial_{n}/train-{context_size}.jsonl", 
#                                             CONTEXT=context_size,
#                                             MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
#                                             GRID_MEMORY="4G")
#             test_set = env.CreateDatapoints(source = train_test_files[env["NUM_TRIALS"]+n], 
#                                             target = f"work/experiments/{exp_type}/trial_{n}/test-{context_size}.jsonl", 
#                                             CONTEXT=context_size,
#                                             MIN_LEN = max(env["CONTEXT_WINDOW_SIZE"]),
#                                             GRID_MEMORY="4G")
#             result, errors = env.TrainModel(source = [train_set, test_set], 
#                                     target = [f"work/experiments/{exp_type}/trial_{n}/result/{context_size}.txt",
#                                                 f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-error-ids.txt"], 
#                                     SAVE_NAME=f"work/experiments/{exp_type}/trial_{n}/best_model/{context_size}.pt",
#                                     CM = f"work/experiments/{exp_type}/confusion-{context_size}.png",
#                                     MODEL="contrastive",
#                                     GRID_GPU_COUNT=1,
#                                     GRID_ACCOUNT="tlippin1_gpu",
#                                     GRID_QUEUE="a100",
#                                     GRID_TIME="45:00")
#             report = env.GenerateReport(source = [errors],
#                                         target = f"work/experiments/{exp_type}/trial_{n}/result/{context_size}-errors.txt",
#                                         PG_PATH = env["PG_FILE"])
#             results.append(result)

