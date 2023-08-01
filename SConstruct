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
    ("DATAPATH", "", "/export/data/english/women_writers.tgz"), # correct
    ("DATAPATH_2", "", "/export/large_corpora/gutenberg/"),
    ("LOCAL_DATA_TAR", "", "./data/local.tar.gz"),
    ("LOCAL_DATA", "", ["./data/warren.adulateur.xml", "./data/haywood.eovaai.xml", "./data/smith.manor.xml"]),
    ("LOCAL_PG", "", "/export/large_corpora/gutenberg/2/7/0/2701/2701-h/2701-h.htm"),
    ("PG_CATALOG", "", "pg_catalog.csv"),
    ("SEGMENT_BY_CH", "", "chapter"),
    ("SEGMENT_BY_PG", "", "paragraph"),
    ("MODEL_NAME", "", "bert-base-uncased"),
    ("MAX_TOKS", "", 512),
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
        "ProcessData" : Builder(
            # action="python scripts/create_data.py --data_path ${SOURCES} --output ${TARGETS} --granularity $SEGMENT_BY_PG",
            action="python scripts/gutenberg.py --base_dir ${DATAPATH_2} --input ${SOURCES} --output ${TARGETS}",
        ),
        "EncodeData": Builder(
            action="python scripts/encode_data.py --input ${SOURCES[0]} --model_name ${MODEL_NAME} --output ${TARGETS} --max_toks ${MAX_TOKS}"
        )
    }
)

# OK, at this point we have defined all the builders and variables, so it's
# time to specify the actual experimental process, which will involve
# running all combinations of datasets, folds, model types, and parameter values,
# collecting the build artifacts from applying the models to test data in a list.
#
# The basic pattern for invoking a build rule is:
#
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"
#
# Note how variables are specified in each invocation, and their values used to fill
# in the build commands *and* determine output filenames.  It's a very flexible system,
# and there are ways to make it less verbose, but in this case explicit is better than
# implicit.
#
# Note also how the outputs ("targets") from earlier invocation are used as the inputs
# ("sources") to later ones, and how some outputs are also gathered into the "results"
# variable, so they can be summarized together after each experiment runs.

# env.ProcessData(source = env["DATA_PATH"] , target = "test.txt")
env.ProcessData(source = env["PG_CATALOG"] , target = "mobydick.txt")
env.encodeData(source = ["result.json"], target = "encoded.h5")