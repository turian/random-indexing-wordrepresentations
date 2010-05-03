#!/usr/bin/env python

import sys
import string
from common.file import myopen
from common.stats import stats
from common.str import percent

import numpy
import random

def trainingsentences():
    """
    For each line (sentence) in the training data, transform it into a list of token IDs.
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
        count += 1
        if count % 1000 == 0:
            print >> sys.stderr, "Read %d lines from training file %s..." % (count, filename)
            print >> sys.stderr, stats()

def generate_context_vectors():
    """
    Generate the (random) context vectors.
    """

    HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
    from vocabulary import wordmap
    assert HYPERPARAMETERS["RANDOMIZATION_TYPE"] == "ternary"

    NONZEROS = int(HYPERPARAMETERS["TERNARY_NON_ZERO_PERCENT"] * HYPERPARAMETERS["REPRESENTATION_SIZE"] + 0.5)

    print >> sys.stderr, "Generating %d nonzeros per %d-length random context vector" % (NONZEROS, HYPERPARAMETERS["REPRESENTATION_SIZE"])

    # Generate one set of context vectors per list in HYPERPARAMETERS["CONTEXT_TYPES"]
    context_vectors = []
    for i in range(len(HYPERPARAMETERS["CONTEXT_TYPES"])):
        print >> sys.stderr, "Generated %s context matrixes" % (percent(i, len(HYPERPARAMETERS["CONTEXT_TYPES"])))
        print >> sys.stderr, stats()
        thiscontext = numpy.zeros((wordmap.len, HYPERPARAMETERS["REPRESENTATION_SIZE"]))
        for j in range(wordmap.len):
            idxs = range(HYPERPARAMETERS["REPRESENTATION_SIZE"])
            random.shuffle(idxs)
            for k in idxs[:NONZEROS]:
                thiscontext[j][k] = random.choice([-1, +1])
#            print thiscontext[j]
        context_vectors.append(thiscontext)

    print >> sys.stderr, "Done generating %s context matrixes" % (percent(i, len(HYPERPARAMETERS["CONTEXT_TYPES"])))
    print >> sys.stderr, stats()
    return context_vectors

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from common import myyaml
    import common.dump
    print >> sys.stderr, myyaml.dump(common.dump.vars_seq([hyperparameters]))

    random.seed(HYPERPARAMETERS["RANDOM_SEED"])
    from vocabulary import wordmap

    context_vectors = generate_context_vectors()

    random_representations = numpy.zeros((wordmap.len, HYPERPARAMETERS["REPRESENTATION_SIZE"]))

    for tokens in trainingsentences():
        for i in range(len(tokens)):
            for j, context in enumerate(HYPERPARAMETERS["CONTEXT_TYPES"]):
                for k in context:
                    tokidx = i + k
                    if tokidx < 0 or tokidx >= len(tokens): continue
                    random_representations[tokens[i]] += context_vectors[j][tokens[tokidx]]
#        print tokens
