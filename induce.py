#!/usr/bin/env python

import logging
import sys
import string
from common.file import myopen
from common.stats import stats
from common.str import percent

import numpy
import random

import diagnostics

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
            logging.info("Read %d lines from training file %s..." % (count, filename))
            logging.info(stats())

def generate_context_vectors():
    """
    Generate the (random) context vectors.
    """

    HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
    from vocabulary import wordmap

    if HYPERPARAMETERS["RANDOMIZATION_TYPE"] == "gaussian":
        context_vectors = [numpy.random.normal(size=(wordmap.len, HYPERPARAMETERS["REPRESENTATION_SIZE"])) for i in range(len(HYPERPARAMETERS["CONTEXT_TYPES"]))]
    elif HYPERPARAMETERS["RANDOMIZATION_TYPE"] == "ternary":
        NONZEROS = int(HYPERPARAMETERS["TERNARY_NON_ZERO_PERCENT"] * HYPERPARAMETERS["REPRESENTATION_SIZE"] + 0.5)
    
        logging.info("Generating %d nonzeros per %d-length random context vector" % (NONZEROS, HYPERPARAMETERS["REPRESENTATION_SIZE"]))
    
        # Generate one set of context vectors per list in HYPERPARAMETERS["CONTEXT_TYPES"]
        context_vectors = []
        for i in range(len(HYPERPARAMETERS["CONTEXT_TYPES"])):
            logging.info("Generated %s context matrixes" % (percent(i, len(HYPERPARAMETERS["CONTEXT_TYPES"]))))
            logging.info(stats())
            thiscontext = numpy.zeros((wordmap.len, HYPERPARAMETERS["REPRESENTATION_SIZE"]))
            for j in range(wordmap.len):
                idxs = range(HYPERPARAMETERS["REPRESENTATION_SIZE"])
                random.shuffle(idxs)
                for k in idxs[:NONZEROS]:
                    thiscontext[j][k] = random.choice([-1, +1])
    #            print thiscontext[j]
            context_vectors.append(thiscontext)
    else:
        assert 0
    
    logging.info("Done generating %s context matrixes" % (percent(i, len(HYPERPARAMETERS["CONTEXT_TYPES"]))))
    logging.info(stats())
    return context_vectors

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)
    import hyperparameters

    from common import myyaml
    import common.dump
    print >> sys.stderr, myyaml.dump(common.dump.vars_seq([hyperparameters]))

    rundir = common.dump.create_canonical_directory(HYPERPARAMETERS)

    import os.path, os
    logfile = os.path.join(rundir, "log")
    if newkeystr != "":
        verboselogfile = os.path.join(rundir, "log%s" % newkeystr)
        print >> sys.stderr, "Logging to %s, and creating link %s" % (logfile, verboselogfile)
        os.system("ln -s log %s " % (verboselogfile))
    else:
        print >> sys.stderr, "Logging to %s, not creating any link because of default settings" % logfile

    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("INITIALIZING TRAINING STATE")
    logging.info(myyaml.dump(common.dump.vars_seq([hyperparameters])))


    import random, numpy
    random.seed(HYPERPARAMETERS["RANDOM_SEED"])
    numpy.random.seed(HYPERPARAMETERS["RANDOM_SEED"])
    from vocabulary import wordmap

    cnt = 0
    random_representations = numpy.zeros((wordmap.len, HYPERPARAMETERS["REPRESENTATION_SIZE"]))

    context_vectors = generate_context_vectors()

    for tokens in trainingsentences():
        for i in range(len(tokens)):
            for j, context in enumerate(HYPERPARAMETERS["CONTEXT_TYPES"]):
                for k in context:
                    tokidx = i + k
                    if tokidx < 0 or tokidx >= len(tokens): continue
                    random_representations[tokens[i]] += context_vectors[j][tokens[tokidx]]
        cnt += 1
        if cnt % 10000 == 0:
            diagnostics.diagnostics(cnt, random_representations)

    logging.info("DONE. Dividing embeddings by their standard deviation...")
    random_representations = random_representations * (1. / numpy.std(random_representations))
    diagnostics.diagnostics(cnt, random_representations)
    diagnostics.visualizedebug(cnt, random_representations, rundir, newkeystr)

    outfile = os.path.join(rundir, "random_representations")
    if newkeystr != "":
        verboseoutfile = os.path.join(rundir, "random_representations%s" % newkeystr)
        logging.info("Writing representations to %s, and creating link %s" % (outfile, verboseoutfile))
        os.system("ln -s random_representations %s " % (verboseoutfile))
    else:
        logging.info("Writing representations to %s, not creating any link because of default settings" % outfile)

    o = open(outfile, "wt")
    from vocabulary import wordmap
    for i in range(wordmap.len):
        o.write(wordmap.str(i) + " ")
        for v in random_representations[i]:
            o.write(`v` + " ")
        o.write("\n")
