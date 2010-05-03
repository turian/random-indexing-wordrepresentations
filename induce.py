#!/usr/bin/env python

import sys
import string
from common.file import myopen

def training_examples():
    """
    An iterator of a list of tokens (one list per line in the input data.)
    """

    HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
    from vocabulary import wordmap
    filename = HYPERPARAMETERS["TRAIN_SENTENCES"]
    count = 0
    for l in myopen(filename):
        tokens = []
        for w in string.split(l):
            w = string.strip(w)
            assert wordmap.exists(w)     # Not exactly clear what to do
                                         # if the word isn't in the vocab.
            tokens.append(wordmap.id(w))
        yield tokens

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from common import myyaml
    import common.dump
    print >> sys.stderr, myyaml.dump(common.dump.vars_seq([hyperparameters]))

    for tokens in training_examples():
        print tokens
