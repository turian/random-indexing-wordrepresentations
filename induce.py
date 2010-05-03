#!/usr/bin/env python

if __name__ == "__main__":
    import common.hyperparameters, common.options
    HYPERPARAMETERS = common.hyperparameters.read("random-indexing")
    HYPERPARAMETERS, options, args, newkeystr = common.options.reparse(HYPERPARAMETERS)

    from common import myyaml
    import sys
    print >> sys.stderr, myyaml.dump(HYPERPARAMETERS)
