"""
Module to update hyperparameters automatically.
"""

from os.path import join
import common.hyperparameters
HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
DATA_DIR = HYPERPARAMETERS["locations"]["DATA_DIR"]
RUN_NAME = HYPERPARAMETERS["RUN_NAME"]
VOCABULARY_SIZE = HYPERPARAMETERS["VOCABULARY_SIZE"]
INCLUDE_UNKNOWN_WORD = HYPERPARAMETERS["INCLUDE_UNKNOWN_WORD"]
HYPERPARAMETERS["TRAIN_SENTENCES"] = join(DATA_DIR, "%s.train.txt.gz" % RUN_NAME)
#HYPERPARAMETERS["ORIGINAL VALIDATION_SENTENCES"] = join(DATA_DIR, "%s.validation.txt.gz" % RUN_NAME)
#HYPERPARAMETERS["VALIDATION_SENTENCES"] = join(DATA_DIR, "%s.validation-%d.txt.gz" % (RUN_NAME, HYPERPARAMETERS["VALIDATION EXAMPLES"]))
HYPERPARAMETERS["VOCABULARY"] = join(DATA_DIR, "vocabulary-%s-%d.txt.gz" % (RUN_NAME, VOCABULARY_SIZE))
HYPERPARAMETERS["VOCABULARY_IDMAP_FILE"] = join(DATA_DIR, "idmap.%s-%d.include_unknown=%s.pkl.gz" % (RUN_NAME, VOCABULARY_SIZE, INCLUDE_UNKNOWN_WORD))

