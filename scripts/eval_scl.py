#!/usr/bin/python
from os.path import join,exists,dirname
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, read_feature_groups, read_feature_lookup
import os
import scipy.sparse
import sys
from sklearn import svm
from sklearn.feature_selection import chi2

def main(args):
    if len(args) < 3:
        sys.stderr.write("Required argument(s): <data file> <pivot file> <matrix file> [backward=False]\n\n")
        sys.exit(-1)

    goal_ind = 2
    (data_file, pivot_file, matrix_file) = args[0:3]
    data_dir = dirname(data_file)
    backward = False

    if len(args) == 4:
        backward = args[3] == 'True'
        sys.stderr.write("Backward set to %s by final argument=%s\n" % (backward, args[3]))

    print("Reading feature group map")
    group_map = read_feature_groups(join(data_dir, 'reduced-feature-groups.txt'))

    print("Reading pivot index file into a dictionary and creating pivot-only and nopivot matrices")
    pivots = read_pivots(pivot_file)
    num_pivots = len(pivots)

    print("Reading input data from %s" % (data_file))
    all_X, all_y = load_svmlight_file(data_file, dtype='float32')
    num_instances, num_feats = all_X.shape
    all_X = all_X.toarray()

    feature_map = read_feature_lookup(join(data_dir, 'reduced-features-lookup.txt'))

    print("  Data has %d instances and %d features" % (num_instances, num_feats))
    #nopivot_X_all = zero_pivot_columns(all_X, pivots)
    #pivot_X_all = zero_nonpivot_columns(all_X, pivots)

    if backward:
        direction = 1
    else:
        direction = 0

    source_domain_ind = group_map['Domain'][direction]
    target_domain_ind = group_map['Domain'][1-direction]
    source_domain = feature_map[source_domain_ind]
    target_domain = feature_map[target_domain_ind]

    print("Evaluating with source domain %s and target domain %s" % (source_domain, target_domain))

    ## Our training data is all labels where the domain feature for our
    ## target domain is set to 0
    train_inds = np.where(all_X[:,target_domain_ind] == 0)[0]
    X_train = all_X[train_inds, :]
    ## After finding the instances, zero out that feature so it isn't used for prediction:
    X_train[:, source_domain_ind] = 0
    X_train[:, target_domain_ind] = 0
    y_train = all_y[train_inds]
    nopivot_X_train = zero_pivot_columns(X_train, pivots)
    pivot_X_train = zero_nonpivot_columns(X_train, pivots)

    ## Our test data is all instances where the domain feature for our
    ## target domain is not 0
    test_inds = np.where(all_X[:,target_domain_ind] > 0)[0]
    X_test = all_X[test_inds,:]
    ## After finding the instances, zero out that feature so it isn't used for prediction:
    X_test[:, source_domain_ind] = 0
    X_test[:, target_domain_ind] = 0
    y_test = all_y[test_inds]

    print("For target domain index %d there are %d training instances and %d test instances" % (target_domain_ind, X_train.shape[0], X_test.shape[0]))

    nopivot_X_test = zero_pivot_columns(X_test, pivots)
    pivot_X_test = zero_nonpivot_columns(X_test, pivots)


    with open(matrix_file, 'rb') as theta_file:
        theta = pickle.load(theta_file)
        num_new_feats = theta.shape[0]


    new_X_train = (theta * nopivot_X_train.transpose()).transpose()
    new_X_test = (theta * nopivot_X_test.transpose()).transpose()

    print("All + new feature space evaluation")
    all_plus_new_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_new_feats), dtype=np.float16))
    all_plus_new_train[:, :num_feats] += X_train
    all_plus_new_train[:, num_feats:] += new_X_train
    all_plus_new_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_new_feats), dtype=np.float16))
    all_plus_new_test[:, :num_feats] += X_test
    all_plus_new_test[:, num_feats:] += new_X_test
    (l2_c, l2_f1) = find_best_c(all_plus_new_train, y_train, pos_label=goal_ind)
    evaluate_and_print_scores(all_plus_new_train, y_train, all_plus_new_test, y_test, goal_ind, l2_c)
    del all_plus_new_train, all_plus_new_test


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
