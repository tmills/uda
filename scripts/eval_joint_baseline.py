#!/usr/bin/python
from os.path import join,exists,dirname
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics import f1_score
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, read_feature_groups, read_feature_lookup
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
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <combined labeled data>\n\n")
        sys.exit(-1)

    goal_ind = 2

    sys.stderr.write("Reading source data from %s\n" % (args[0]))
    all_X, all_y = load_svmlight_file(args[0])
    num_instances, num_feats = all_X.shape

    domain_map = read_feature_groups(join(dirname(args[0]), 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(dirname(args[0]), 'reduced-features-lookup.txt'))

    for direction in [0,1]:
        score_ave = 0
        sys.stderr.write("using domain %s as source, %s as target\n"  %
            (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

        train_instance_inds = np.where(all_X[:,domain_inds[direction]].toarray() > 0)[0]
        X_train = all_X[train_instance_inds,:]
        X_train[:, domain_inds[direction]] = 0
        X_train[:, domain_inds[1-direction]] = 0
        y_train = all_y[train_instance_inds]
        num_train_instances = X_train.shape[0]

        test_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
        X_test = all_X[test_instance_inds,:]
        X_test[:, domain_inds[direction]] = 0
        X_test[:, domain_inds[1-direction]] = 0
        y_test = all_y[test_instance_inds]
        num_test_instances = X_test.shape[0]

        (l2_c, l2_f1) = find_best_c(X_test, y_test, scorer=f1_score, pos_label=goal_ind)
        print("Target-target f1 score after tuning is %f" % (l2_f1))

        for fold in [0,1]:
            split_ind = num_test_instances // 2
            ## Split the test data in half and join with source:
            if fold == 0:
                target_train_X = X_test[:split_ind,:]
                target_train_y = y_test[:split_ind]
                target_test_X = X_test[split_ind:,:]
                target_test_y = y_test[split_ind:]
            else:
                target_test_X = target_train_X
                target_test_y = target_train_y
                target_train_X = X_test[split_ind:, :]
                target_train_y = y_test[split_ind:]

            joint_train_X = np.zeros((num_train_instances+target_train_X.shape[0], num_feats))
            joint_train_X[:num_train_instances,:] += X_train
            joint_train_X[num_train_instances:,:] += target_train_X
            joint_train_y = np.zeros(num_train_instances+target_train_X.shape[0])
            joint_train_y[:num_train_instances] += y_train
            joint_train_y[num_train_instances:] += target_train_y

            (l2_c, l2_f1) = find_best_c(joint_train_X, joint_train_y, scorer=f1_score, pos_label=goal_ind)
            print("In fold %d, the best c=%f led to f1=%f" % (fold+1, l2_c, l2_f1))
            score_ave += l2_f1 / 2.0

        print("Average f1 score of source+target-target evaluation: %f" % score_ave)

if __name__ == "__main__":
    main(sys.argv[1:])
