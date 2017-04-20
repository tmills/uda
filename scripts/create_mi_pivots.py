#!/usr/bin/env python

import sys
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import mutual_info_classif as mi
import uda_common
import numpy as np
import scipy.sparse

def main(args):
    if len(args) < 2:
        sys.stderr.write("Error: Two required arguments: <source liblinear> <target liblinear>\n")
        sys.exit(-1)

    source_file, target_file = args
    X_train, y_train = load_svmlight_file(source_file, dtype='float32')
    X_test, y_test = load_svmlight_file(target_file, dtype='float32')
    num_instances, num_feats = X_train.shape
    num_test_instances, num_test_feats = X_test.shape
    X_test = uda_common.align_test_X_train(X_train, X_test)

    mi_label = mi(X_train, y_train)

    y_corpus = np.zeros(num_instances+num_test_instances)
    y_corpus[0:num_instances] = 1
    X_combined = scipy.sparse.lil_matrix((len(y_corpus), num_feats))
    X_combined[:num_instances] += X_train
    X_combined[num_instances:] += X_test

    mi_domains = mi(X_combined, y_corpus)

    ## mi is between 0 (no information) and 1 (perfect information)
    ## we want the label informatio to be high (~1)
    ## we want the domain information to be low (1-mi ~ 1)
    ## we want both these things to be true so we use *
    mi_joint = mi_label * (1 - mi_domains)

    ## I want high values, so I reverse the list and sort
    ranked_inds = np.argsort(1 - mi_joint)
    pivots = np.sort(ranked_inds[:100])

    for pivot in pivots:
        print(pivot)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
