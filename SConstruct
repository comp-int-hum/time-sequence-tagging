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
import steamroller

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("custom.py")
vars.AddVariables(
    ("WW_DATAPATH", "", "/export/data/english/women_writers.tgz"), # correct
    ("PG_DATAPATH", "", "/export/large_corpora/gutenberg/"),
    ("LOCAL_DATA", "", ["./data/warren.adulateur.xml", "./data/haywood.eovaai.xml", "./data/smith.manor.xml"]),
    ("PG_CATALOG", "", "pg_catalog.csv"),
    ("ENC_MODEL_NAME", "", "bert-base-uncased"),
    ("MAX_TOKS", "", 512),
    ("LOCAL", "", "False"),
    ("DATA_SIZE", "Number of texts to encode", 3000),
    ("SAMPLE_SIZE", "Number of texts to grab chapters from", 625),
    ("TRAIN_TEST_SPLIT", "", 0.8),
    ("SAMPLES", "Number of chapters to sample from text", 5),
    ("EMB_DIM", "Size of BERT embedding", 1536),
    ("EPOCHS", "", 50),
    ("SAME_CH", "", "True"),
    ("CH_EMBED_TYPE", "", ["inc_fl", "only_fl", "no_fl"]),
    ("CD", "", "cd"), # Cross_domain: cd or no_cd
    ("NUM_TRIALS", "", 20),
    ("MODEL_EVAL", "", "True"),
    ("SAMPLE_SEED", "", 1),
    ("DPS_SEED", "", 1)
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],
    
    # Defining a bunch of builders (none of these do anything except "touch" their targets,
    # as you can see in the dummy.py script).  Consider in particular the "TrainModel" builder,
    # which interpolates two variables beyond the standard SOURCES/TARGETS: PARAMETER_VALUE
    # and MODEL_TYPE.  When we invoke the TrainModel builder (see below), we'll need to pass
    # in values for these (note that e.g. the existence of a MODEL_TYPES variable above doesn't
    # automatically populate MODEL_TYPE, we'll do this with for-loops).
    BUILDERS={
        "ProcessPGLocal" : Builder(
            # action="python scripts/create_data.py --data_path ${SOURCES} --output ${TARGETS} --granularity $SEGMENT_BY_PG",
            action="python scripts/process_pg.py --base_dir ${LOCAL_TEST} --input ${SOURCES} --output ${TARGETS} --local $LOCAL",
        ),
        "ProcessPG" : Builder(
            # action="python scripts/create_data.py --data_path ${SOURCES} --output ${TARGETS} --granularity $SEGMENT_BY_PG",
            action="python scripts/process_pg.py --base_dir ${PG_DATAPATH} --input ${SOURCES} --output ${TARGETS} --local ${LOCAL}",
        ),
        "ProcessWW" : Builder(
            action="python scripts/process_ww.py --output ${TARGETS} --data_path ${SOURCES}",
        ),
        "CreateSample": Builder (
            action="python scripts/create_sample.py --inputs ${SOURCES} --output ${TARGETS[0]} --data_size ${DATA_SIZE} --seed {SAMPLE_SEED}"
        ),
        "EncodeData": Builder(
            action="python scripts/encode_data.py --input ${SOURCES[0]} --model_name ${ENC_MODEL_NAME} --output ${TARGETS} --max_toks ${MAX_TOKS}"
        ),
        "ShuffleData": Builder(
            action="python scripts/shuffle_data.py --inputs ${SOURCES} --output ${TARGETS} --sample_size ${SAMPLE_SIZE} --split_ratio ${TRAIN_TEST_SPLIT} --cd ${CROSS_DOMAIN} --seed ${SHUFFLE_SEED}"
        ),
        "CreateDatapoints": Builder(
            action="python scripts/create_datapoints.py --input ${SOURCES} --output ${TARGETS} --samples ${SAMPLES} --same ${SAME_CH} --fl ${FL} --seed ${DPS_SEED}"
        ),
        "TrainModel": Builder(
            action="python scripts/train_model.py --train ${SOURCES[0]} --test ${SOURCES[1]} --model_name ${SAVE_NAME} --emb_dim ${EMB_DIM} --num_epochs ${EPOCHS} --result ${TARGETS} --cum ${CUM}"
        )
    }
)

# Get data
if env["LOCAL"] == "True":
    print("Debugging run")
    pg_local = env.ProcessPGLocal(source = env["PG_CATALOG"] , target = ["work/gutenberg.jsonl", "work/test.txt"])
else:
    print("Gathering all data")
    pg_data = env.ProcessPG(source = env["PG_CATALOG"] , target = ["work/data/gutenberg.jsonl"])
    ww_data = env.ProcessWW(source = env["WW_DATAPATH"], target = ["work/data/womenwriters.jsonl"])

# Downsample Step
pg_sample = env.CreateSample(source = [pg_data], target = ["work/data/pg_sample.jsonl"])
ww_sample = env.CreateSample(source = [pg_data], target = ["work/data/ww_sample.jsonl"])

# Encode Step
pg_enc = env.EncodeData(source = pg_sample, target = f"work/data/pg_encoded.jsonl")
ww_enc = env.EncodeData(source = ww_sample, target = f"work/data/ww_encoded.jsonl")

# Shuffle + Split
for i in range(env["NUM_TRIALS"]):
    if env["CD"] == "cd":
        print(f"Cross domain data shuffle")
        train_data, test_data = env.ShuffleData(source = [pg_enc], 
                                                target = [f"work/experiments/{env['CD']}/trial_{i}/shuffle_train.jsonl",
                                                          f"work/{env['CD']}/trial_{i}/shuffled_test.jsonl"], 
                                                CROSS_DOMAIN = ww_enc,
                                                SHUFFLE_SEED = i)
    else:
        print(f"Within domain data shuffle")
        train_data, test_data = env.ShuffleData(source = [pg_enc], 
                                                target = [f"work/experiments/{env['CD']}/trial_{i}/shuffle_train.jsonl",
                                                          f"work/{env['CD']}/trial_{i}/shuffled_test.jsonl"],
                                                SHUFFLE_SEED = i)

    for fl_type in env["CH_EMBED_TYPE"]:
        print(f"Fl type: {fl_type}")
        train_set = env.CreateDatapoints(source = train_data, 
                                         target = f"work/experiments/{env['CD']}/trial_{i}/train-{fl_type}.jsonl", 
                                         FL=fl_type)
        test_set = env.CreateDatapoints(source = test_data, 
                                        target = f"work/experiments/{env['CD']}/trial_{i}/test-{fl_type}.jsonl", 
                                        FL=fl_type)
        result = env.TrainModel(source = [train_set, test_set], 
                                target = f"work/experiments/{env['CD']}/trial_{i}/result/{fl_type}.txt", 
                                SAVE_NAME=f"work/experiments/{env['CD']}/trial_{i}/best_model/{fl_type}.pt",
                                CUM = f"work/experiments/{env['CD']}/cum-{fl_type}.txt")
        
