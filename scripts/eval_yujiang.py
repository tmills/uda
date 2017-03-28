#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import evaluate_and_print_scores
import numpy as np
import scipy.sparse
import sys

def main(args):
    if len(args) < 2:
        sys.stderr.write("Two required arguments: <source file> <target file>\n")
        sys.exit(-1)

    source_file, target_file = args

    print("Reading input data from %s" % (source_file))
    X_train, y_train = load_svmlight_file(source_file)
    num_instances, num_feats = X_train.shape
    X_test, y_test = load_svmlight_file(target_file)
    num_test_instances, num_test_feats = X_test.shape
    if num_test_feats < num_feats:
        ## Expand X_test
        #print("Not sure I need to do anything here.")
        X_test_array = X_test.toarray()
        X_test = scipy.sparse.csr_matrix(np.append(X_test_array, np.zeros((num_test_instances, num_feats-num_test_feats)), axis=1))
    elif num_test_feats > num_feats:
        ## Truncate X_test
        X_test = X_test[:,:num_feats]

    print("Yu and Jiang method (50 similarity features)")
    num_exemplars = 50
    indices = np.sort(np.random.choice(num_test_instances, num_exemplars, replace=False))
    test_exemplars = X_test[indices]
    ## Normalize
    test_exemplars /= test_exemplars.sum(1)
    ## Create a new feature for every train instance that is the similarity with each of these exemplars:
    ## Output matrix is num_train_instances x num_exemplars. add this to end of X_train:
    similarity_features_train = X_train * test_exemplars.transpose()
    similarity_features_test = X_test * test_exemplars.transpose()
    all_plus_sim_X_train = np.matrix(np.zeros((num_instances, num_feats + num_exemplars)))
    all_plus_sim_X_train[:, :num_feats] += X_train
    all_plus_sim_X_train[:, num_feats:] += similarity_features_train
    all_plus_sim_X_test = np.matrix(np.zeros((num_test_instances, num_feats + num_exemplars)))
    all_plus_sim_X_test[:, :num_feats] += X_test
    all_plus_sim_X_test[:,num_feats:] += similarity_features_test
    evaluate_and_print_scores(all_plus_sim_X_train, y_train, all_plus_sim_X_test, y_test, 2)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
