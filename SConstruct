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
    ("LOCAL_DATA_TAR", "", "./data/local.tar.gz"),
    ("LOCAL_DATA", "", ["./data/warren.adulateur.xml", "./data/haywood.eovaai.xml", "./data/smith.manor.xml"]),
    ("PG_CATALOG", "", "pg_catalog.csv"),
    ("SEGMENT_BY_CH", "", "chapter"),
    ("SEGMENT_BY_PG", "", "paragraph"),
    ("MODEL_NAME", "", "bert-base-uncased"),
    ("MAX_TOKS", "", 512),
    ("LOCAL", "", "False"),
    ("LOCAL_TEST", "", "./test/"),
    ("DATA_SIZE", "Number of texts to grab chapters from", 625),
    ("TRAIN_TEST_SPLIT", "", 0.8),
    ("SAMPLES", "", 5),
    ("EMB_DIM", "", 1536),
    ("EPOCHS", "", 50),
    ("DIFF_CH", "", True),
    ("CH_EMBED_TYPE", "", ["only_fl", "no_fl", "inc_fl"])
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
        "ShuffleData": Builder(
            action="python scripts/shuffle_data.py --input ${SOURCES} --output ${TARGETS} --data_size ${DATA_SIZE} --split_ratio ${TRAIN_TEST_SPLIT}"
        ),
        "EncodeData": Builder(
            action="python scripts/encode_data.py --input ${SOURCES[0]} --model_name ${MODEL_NAME} --output ${TARGETS} --max_toks ${MAX_TOKS}"
        ),
        "CreateDatapoints": Builder(
            action="python scripts/create_datapoints.py --input ${SOURCES} --output ${TARGETS} --samples ${SAMPLES} --same ${DIFF_CH} --fl ${FL}"
        ),
        "TrainModel": Builder(
            action="python scripts/train_model.py --train ${SOURCES[0]} --eval ${SOURCES[1]} --model_name ${SAVE_NAME} --emb_dim ${EMB_DIM} --num_epochs ${EPOCHS} --result ${TARGETS}"
        )
    }
)


# TODO: Fix this. It's horrific. I'm sorry.

if env["LOCAL"] == "True":
    print("Is local")
    data = env.ProcessPGLocal(source = env["PG_CATALOG"] , target = ["work/gutenberg.jsonl", "work/test.txt"])
else:
    data = env.ProcessPG(source = env["PG_CATALOG"] , target = ["work/gutenberg.jsonl", "work/test.txt"])
    print("Is not local")

train, test = env.ShuffleData(source = data[0], target = ["work/shuffled_gb_train.jsonl", "work/shuffled_gb_test.jsonl"])
train_enc = env.EncodeData(source = train, target = "work/train_encoded.jsonl")
test_enc = env.EncodeData(source = test, target = "work/test_encoded.jsonl")

for fl_type in env["CH_EMBED_TYPE"]:
    train_data = env.CreateDatapoints(source = train_enc, target = f"work/train-{fl_type}.jsonl", FL=fl_type)
    test_data = env.CreateDatapoints(source = test_enc, target = f"work/test.jsonl-{fl_type}", FL=fl_type)
    result = env.TrainModel(source = [train_data, test_data], target = f"work/result/{fl_type}.txt", SAVE_NAME=f"work/best_model/{fl_type}.pt")