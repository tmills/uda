#!/usr/bin/python
from os.path import join,exists
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics import f1_score
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c
import os
import scipy.sparse
import sys
from sklearn import svm
from sklearn.feature_selection import chi2

## This script gets a baseline for domain adaptation based on a combined training
## set containing source and target training data. This is a better ceiling for
## adaptation performance than source-source or target-target evaluations.
## This way, if guidelines are different, the discriminating line p(y|x) will
## be different, and performance will be lower than target-target.
## Since we have been using target _trainign_ set for testing (for greater power)
## we have a problem adding it because we can't have it be part of training and
## test set. So what I do is basically 2-fold experiments where half the target
## training data is added to the source, test on the other half, and then reverse
## and calculate again.
def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <labeled source data> <labeled target data>\n\n")
        sys.exit(-1)

    goal_ind = 2
    (source_file, target_file) = args

    sys.stderr.write("Reading source data from %s\n" % (source_file))
    X_train, y_train = load_svmlight_file(source_file, dtype='float32')
    num_instances, num_feats = X_train.shape

    sys.stderr.write("Reading target data from %s\n" % (target_file))
    X_test, y_test = load_svmlight_file(target_file)
    X_test = align_test_X_train(X_train, X_test)
    num_test_instances = X_test.shape[0]

    (l2_c, l2_f1) = find_best_c(X_test, y_test, scorer=f1_score, pos_label=goal_ind)
    print("Target-target score after tuning is %f" % (l2_f1))

    score_ave = 0
    for split_part in [0,1]:
        split_ind = num_test_instances // 2

        ## Split the test data in half and join with source:
        if split_part == 0:
            target_train_X = X_test[:split_ind,:]
            target_train_y = y_test[:split_ind]
            target_test_X = X_test[split_ind:,:]
            target_test_y = y_test[split_ind:]
        else:
            target_test_X = target_train_X
            target_test_y = target_train_y
            target_train_X = X_test[split_ind:, :]
            target_train_y = y_test[split_ind:]

        joint_train_X = np.zeros((num_instances+target_train_X.shape[0], num_feats))
        joint_train_X[:num_instances,:] += X_train
        joint_train_X[num_instances:,:] += target_train_X
        joint_train_y = np.zeros(num_instances+target_train_X.shape[0])
        joint_train_y[:num_instances] += y_train
        joint_train_y[num_instances:] += target_train_y

        (l2_c, l2_f1) = find_best_c(joint_train_X, joint_train_y, scorer=f1_score, pos_label=goal_ind)
        print("In fold %d, the best c=%f led to f1=%f" % (split_part+1, l2_c, l2_f1))
        score_ave += l2_f1 / 2.0

    print("Average f1 score of source+target-target evaluation: %f" % score_ave)

if __name__ == "__main__":
    main(sys.argv[1:])
