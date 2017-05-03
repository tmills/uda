#!/usr/bin/python
import numpy as np
import os
from numpy.linalg import pinv
from scipy.linalg import svd
import scipy.sparse
import sys

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, remove_columns

def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <labeled source data> <labeled target data>\n\n")
        sys.exit(-1)

    goal_ind = 2
    freq_cutoff = 5
    (source_file, target_file) = args

    print("Reading input data from %s" % (source_file))
    X_train, y_train = load_svmlight_file(source_file, dtype='float64')
    num_instances, num_feats = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))

    X_test, y_test = load_svmlight_file(target_file)
    X_test = align_test_X_train(X_train, X_test)
    num_test_instances = X_test.shape[0]

    print("Sun et al. 15 UFEDA")
    ## Start by removing bias term that liblinear puts in -- it will always be 1 so has no
    ## variance with other variables, and the zero causes numerical issues later.
    to_remove = [0]
    train_feat_counts = X_train.sum(0)
    ranked_inds = np.argsort(train_feat_counts)
    for i in range(ranked_inds.shape[1]):
        ind = ranked_inds[0,i]
        if train_feat_counts[0,ind] <= freq_cutoff:
            to_remove.append(ind)
        else:
            break

    X_train_smaller = remove_columns(X_train.toarray(), to_remove)
    X_test_smaller = remove_columns(X_test.toarray(), to_remove)

    print("Removed %d features that are infrequent in training data and reduced shape is %s" % (len(to_remove), str(X_train_smaller.shape)))

    to_remove = [0]
    test_feat_counts = X_test_smaller.sum(0)
    test_ranked_inds = np.argsort(test_feat_counts)
    for i in range(test_ranked_inds.shape[1]):
        ind = test_ranked_inds[0,i]
        if test_feat_counts[0,ind] <= freq_cutoff:
            to_remove.append(ind)
        else:
            break

    X_train_smaller = remove_columns(X_train_smaller.toarray(), to_remove)
    X_test_smaller = remove_columns(X_test_smaller.toarray(), to_remove)
    print("Removed %d additional features that are infrequent in test data and reduced shape is %s" % (len(to_remove), str(X_train_smaller.shape)))

    ## Center them on the training data mean:
    X_train_smaller = scipy.sparse.csr_matrix(X_train_smaller - X_train_smaller.mean(0))
    X_test_smaller = scipy.sparse.csr_matrix(X_test_smaller - X_test_smaller.mean(0))

    (l2_c, l2_f1) = find_best_c(X_train_smaller, y_train, goal_ind)
    evaluate_and_print_scores(X_train_smaller, y_train, X_test_smaller, y_test, 2, l2_c)

    cov_train = np.cov(X_train_smaller.toarray(), rowvar=False) + np.eye(X_train_smaller.shape[1])
    U_s, sigma_s, U_s_T = svd(cov_train)
    sigma_s[np.where(sigma_s < 0)] = 0
    sigma_inv_s = pinv(sigma_s * np.eye(len(sigma_s)))
    cov_test = np.cov(X_test_smaller.toarray(), rowvar=False) + np.eye(X_test_smaller.shape[1])
    U_t, sigma_t, U_t_T = svd(cov_test)
    sigma_t[np.where(sigma_t < 0)] = 0
    sigma_diag_t = sigma_t * np.eye(len(sigma_t))

    r = min(np.linalg.matrix_rank(cov_train), np.linalg.matrix_rank(cov_test))
    A_first = np.matrix(U_s) * np.matrix(np.sqrt(sigma_inv_s)) * np.matrix(U_s_T)
    A_second = np.matrix(U_t[:r,:r]) * np.matrix(np.sqrt(sigma_diag_t[:r,:r])) * np.matrix(U_t_T[:r,:r])
    A = A_first * A_second
    recolored_train = X_train_smaller * A
    #whitened_train = scipy.sparse.csr_matrix(X_train_smaller * cov_train**-0.5)
    #recolored_train = whitened_train * (cov_test.transpose()**0.5)
    (l2_c, l2_f1) = find_best_c(recolored_train, y_train, goal_ind)
    evaluate_and_print_scores(recolored_train, y_train, X_test_smaller, y_test, 2, l2_c)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
