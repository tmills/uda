#!/usr/bin/env python

import numpy as np
from sklearn.datasets import load_svmlight_file

import sys

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error: One reqiured argument: <reduced dataset>\n")
        sys.exit(-1)

    data_file = args[0]

    ## load the data:
    all_X, all_y = load_svmlight_file(data_file)
    num_instances, num_feats = all_X.shape

    pivots = np.random.choice(num_feats, 100, replace=False)
    pivots.sort()

    for pivot in pivots:
        print(pivot)

if __name__ == '__main__':
    main(sys.argv[1:])
