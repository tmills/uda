#!/usr/bin/python
from os.path import join,exists
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c
import os
import scipy.sparse
import sys
from sklearn import svm
from sklearn.feature_selection import chi2

def main(args):
    if len(args) < 4:
        sys.stderr.write("Required argument(s): <labeled source data> <labeled target data> <matrix directory> <pivot index file>\n\n")
        sys.exit(-1)

    goal_ind = 2
    (source_file, target_file, matrix_dir, pivot_file) = args

    print("Reading pivot index file into a dictionary and creating pivot-only and nopivot matrices")
    pivots = read_pivots(pivot_file)
    num_pivots = len(pivots)

    print("Reading input data from %s" % (source_file))
    X_train, y_train = load_svmlight_file(source_file, dtype='float32')
    num_instances, num_feats = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))
    nopivot_X_train = zero_pivot_columns(X_train, pivots)
    pivot_X_train = zero_nonpivot_columns(X_train, pivots)

    X_test, y_test = load_svmlight_file(target_file)
    X_test = align_test_X_train(X_train, X_test)
    num_test_instances = X_test.shape[0]

    nopivot_X_test = zero_pivot_columns(X_test, pivots)
    pivot_X_test = zero_nonpivot_columns(X_test, pivots)

    print("Original feature space evaluation (AKA no adaptation, AKA pivot+non-pivot)")
    ## C < 1 => more regularization
    ## C > 1 => more fitting to training
    C_list = [0.001, 0.01, 0.1, 1.0, 10.0]
    (l1_c, l1_f1) = find_best_c(X_train, y_train, goal_ind, penalty='l1', dual=False)
    print("Optimizing l1 with cross-validation gives C=%f and f1=%f" % (l1_c, l1_f1))
    (l2_c, l2_f1) = find_best_c(X_train, y_train, goal_ind)
    print("Optimizing l2 with cross-validation gives C=%f and f1=%f" % (l2_c, l2_f1))

    with open(join(matrix_dir, 'theta_svd.pkl'), 'rb') as theta_file:
        theta = pickle.load(theta_file)
        num_new_feats = theta.shape[1]

    with open(join(matrix_dir, 'theta_full.pkl'), 'rb') as theta_file:
        theta_full = pickle.load(theta_file)
        num_pivots = theta_full.shape[1]

    new_X_train = nopivot_X_train * theta
    new_X_test = nopivot_X_test * theta

    pivotpred_X_train = nopivot_X_train * theta_full
    pivotpred_X_test = nopivot_X_test * theta_full

    print("All + new feature space evaluation")
    all_plus_new_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_new_feats)))
    all_plus_new_train[:, :num_feats] += X_train
    all_plus_new_train[:, num_feats:] += new_X_train
    all_plus_new_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_new_feats)))
    all_plus_new_test[:, :num_feats] += X_test
    all_plus_new_test[:, num_feats:] += new_X_test
    (l2_c, l2_f1) = find_best_c(all_plus_new_train, y_train, goal_ind)
    evaluate_and_print_scores(all_plus_new_train, y_train, all_plus_new_test, y_test, goal_ind, l2_c)
    del all_plus_new_train, all_plus_new_test

    print("Pivot + new feature space evaluation")
    pivot_plus_new_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_new_feats)))
    pivot_plus_new_train[:, :num_feats] += pivot_X_train
    pivot_plus_new_train[:, num_feats:] += new_X_train
    pivot_plus_new_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_new_feats)))
    pivot_plus_new_test[:, :num_feats] += pivot_X_test
    pivot_plus_new_test[:, num_feats:] += new_X_test
    (l2_c, l2_f1)=  find_best_c(pivot_plus_new_train, y_train, goal_ind)
    evaluate_and_print_scores(pivot_plus_new_train, y_train, pivot_plus_new_test, y_test, goal_ind, l2_c)
    del pivot_plus_new_train, pivot_plus_new_test

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
