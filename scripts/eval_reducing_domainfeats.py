#!/usr/bin/python
from os.path import join,exists,dirname
import numpy as np
from numpy import argsort
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, read_feature_groups, read_feature_lookup
import os
import scipy.sparse
import sys
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import chi2

def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <data file> <backward=False>\n\n")
        sys.exit(-1)

    penalty='l2'
    dual=False

    goal_ind = 2
    data_file = args[0]
    data_dir = dirname(data_file)
    backward = args[1] == 'True'
    sys.stderr.write("Backward set to %s by argument=%s\n" % (backward, args[1]))

    print("Reading feature group map")
    group_map = read_feature_groups(join(data_dir, 'reduced-feature-groups.txt'))

    print("Reading input data from %s" % (data_file))
    all_X, all_y = load_svmlight_file(data_file, dtype='float32')
    num_instances, num_feats = all_X.shape
    all_X = all_X.toarray()

    feature_map = read_feature_lookup(join(data_dir, 'reduced-features-lookup.txt'))
    print("  Data has %d instances and %d features" % (num_instances, num_feats))

    if backward:
        direction = 1
    else:
        direction = 0
    
    ## Organize the data matrices:
    source_domain_ind = group_map['Domain'][direction]
    target_domain_ind = group_map['Domain'][1-direction]
    source_domain = feature_map[source_domain_ind]
    target_domain = feature_map[target_domain_ind]
    source_inds = np.where(all_X[:,target_domain_ind] == 0)[0]
    target_inds = np.where(all_X[:,source_domain_ind] == 0)[0]
    y_domain = np.zeros(num_instances)
    y_domain[target_inds] = 1
    # zero out domain features
    all_X[:,source_domain_ind] = 0
    all_X[:,target_domain_ind] = 0
    X_train = all_X[source_inds, :]
    y_train = all_y[source_inds]
    X_test = all_X[target_inds, :]
    y_test = all_y[target_inds]
    # baseline based on always choosing the most common class:
    freq_acc_baseline = max( y_domain.sum() / len(y_domain), 1-(y_domain.sum()) / len(y_domain))

    ## Get the current task performance:
    task_c, _ = find_best_c(X_train, y_train, pos_label=goal_ind)
    ## Get the current domain separation performance:
    scorer = accuracy_score
    _, domain_acc = find_best_c(all_X, y_domain, scorer=scorer)

    print("Evaluating with source domain %s and target domain %s" % (source_domain, target_domain))
    print("Before starting: ")
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, goal_ind, task_c)
    print("F score for domain separation is %f" % (domain_acc))

    # Get ordered features by abs(chi^2):
    chi2_scores, pval = chi2(all_X, y_domain)
    ordered_feats = argsort(np.abs(chi2_scores))
    ind = -1
    # Remove one feature at a time based on how good it as at distinguishing the two domains:
    while (domain_acc - freq_acc_baseline) > 0.1:
        print("*** Next iteration ***")
        removable_feat_ind = ordered_feats[ind]
        all_X[:,removable_feat_ind] = 0
        ind -= 1
        print("Removing feature index %d which had chi2 score of %s" % (removable_feat_ind, str(chi2_scores[removable_feat_ind])))
        _, domain_acc = find_best_c(all_X, y_domain, scorer=scorer)

        task_c, _ = find_best_c(X_train, y_train, pos_label=goal_ind)
        evaluate_and_print_scores(X_train, y_train, X_test, y_test, score_label=goal_ind, C=task_c)
        print("F score for domain separation is %f" % (domain_acc))

if __name__ == "__main__":
    main(sys.argv[1:])
