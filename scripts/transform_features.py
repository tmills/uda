#!/usr/bin/python
from os.path import join,exists
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import remove_pivot_columns, remove_nonpivot_columns, read_pivots, evaluate_and_print_scores
import os
import scipy.sparse
import sys

def main(args):
    if len(args) < 4:
        sys.stderr.write("Required argument(s): <labeled source data> <labeled target data> <matrix directory> <pivot index file>\n\n")
        sys.exit(-1)

    (source_file, target_file, matrix_dir, pivot_file) = args

    print("Reading pivot index file into a dictionary and creating pivot-only and nopivot matrices")
    pivots = read_pivots(pivot_file)
    num_pivots = len(pivots)

    print("Reading input data from %s" % (source_file))
    X_train, y_train = load_svmlight_file(source_file)
    num_instances, num_feats = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))
    nopivot_X_train = remove_pivot_columns(X_train, pivots)
    pivot_X_train = remove_nonpivot_columns(X_train, pivots)

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

    nopivot_X_test = remove_pivot_columns(X_test, pivots)
    pivot_X_test = remove_nonpivot_columns(X_test, pivots)

    print("Original feature space evaluation (AKA no adaptation, AKA pivot+non-pivot)")
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, 2)

    with open(join(args[2], 'theta_svd.pkl'), 'rb') as theta_file:
        theta = pickle.load(theta_file)
        num_new_feats = theta.shape[1]

    with open(join(args[2], 'theta_full.pkl'), 'rb') as theta_file:
        theta_full = pickle.load(theta_file)
        num_pivots = theta_full.shape[1]

    print("Pivot-only feature space evaluation")
    evaluate_and_print_scores(pivot_X_train, y_train, pivot_X_test, y_test, 2)

    print("Non-pivot only feature space evaluation")
    evaluate_and_print_scores(nopivot_X_train, y_train, nopivot_X_test, y_test, 2)

    print("New-only feature space evaluation (svd)")
    new_X_train = nopivot_X_train * theta
    new_X_test = nopivot_X_test * theta
    evaluate_and_print_scores(new_X_train, y_train, new_X_test, y_test, 2)

    print("New-only features space evaluation (no svd)")
    pivotpred_X_train = nopivot_X_train * theta_full
    pivotpred_X_test = nopivot_X_test * theta_full
    evaluate_and_print_scores(pivotpred_X_train, y_train, pivotpred_X_test, y_test, 2)
    #del pivotpred_X_train
    #del pivotpred_X_test

    print("All + new feature space evaluation")
    all_plus_new_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_new_feats)))
    all_plus_new_train[:, :num_feats] += X_train
    all_plus_new_train[:, num_feats:] += new_X_train
    all_plus_new_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_new_feats)))
    all_plus_new_test[:, :num_feats] += X_test
    all_plus_new_test[:, num_feats:] += new_X_test
    evaluate_and_print_scores(all_plus_new_train, y_train, all_plus_new_test, y_test, 2)
    del all_plus_new_train, all_plus_new_test

    print("All + no-svd pivot feature space")
    all_plus_pivotpred_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_pivots)))
    all_plus_pivotpred_train[:, :num_feats] += X_train
    all_plus_pivotpred_train[:, num_feats:] += pivotpred_X_train
    all_plus_pivotpred_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_pivots)))
    all_plus_pivotpred_test[:, :num_feats] += X_test
    all_plus_pivotpred_test[:, num_feats:] += pivotpred_X_test
    evaluate_and_print_scores(all_plus_pivotpred_train, y_train, all_plus_pivotpred_test, y_test, 2)
    del all_plus_pivotpred_train, all_plus_pivotpred_test

    print("Pivot + new feature space evaluation")
    pivot_plus_new_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_new_feats)))
    pivot_plus_new_train[:, :num_feats] += pivot_X_train
    pivot_plus_new_train[:, num_feats:] += new_X_train
    pivot_plus_new_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_new_feats)))
    pivot_plus_new_test[:, :num_feats] += pivot_X_test
    pivot_plus_new_test[:, num_feats:] += new_X_test
    evaluate_and_print_scores(pivot_plus_new_train, y_train, pivot_plus_new_test, y_test, 2)
    del pivot_plus_new_train, pivot_plus_new_test

    print("Pivot + pivot prediction space")
    pivot_plus_pivot_pred_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_pivots)))
    pivot_plus_pivot_pred_train[:,:num_feats] += pivot_X_train
    pivot_plus_pivot_pred_train[:, num_feats:] += pivotpred_X_train
    pivot_plus_pivot_pred_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_pivots)))
    pivot_plus_pivot_pred_test[:, :num_feats] += pivot_X_test
    pivot_plus_pivot_pred_test[:, num_feats:] += pivotpred_X_test
    evaluate_and_print_scores(pivot_plus_pivot_pred_train, y_train, pivot_plus_pivot_pred_test, y_test, 2)
    del pivot_plus_pivot_pred_train, pivot_plus_pivot_pred_test

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
