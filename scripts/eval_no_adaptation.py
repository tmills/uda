#!/usr/bin/python
from os.path import join,exists,dirname
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics import f1_score, precision_recall_fscore_support
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, read_feature_groups, read_feature_lookup
import os
import scipy.sparse
import sys
from sklearn import svm

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <combined & reduced labeled data>\n\n")
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
        y_train = all_y[train_instance_inds]
        num_train_instances = X_train.shape[0]

        test_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
        X_test = all_X[test_instance_inds,:]
        y_test = all_y[test_instance_inds]
        num_test_instances = X_test.shape[0]

        (l2_c, l2_f1) = find_best_c(X_train, y_train, scorer=f1_score, pos_label=goal_ind)
        print("Tuning on source selected c value %f" % (l2_c))

        clf = svm.LinearSVC(C=l2_c)
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        # f1 = f1_score(y_test, y_predicted, pos_label=goal_ind)
        p,r,f1,_ = precision_recall_fscore_support(y_test, y_predicted, average='binary', pos_label=goal_ind)
        print("Precision,recall,F-score on test data is %f\t%f\t%f" % (p,r,f1))

        best_c = best_f = best_r = best_p = 0
        for c_exp in range(-3, 4):
            c = 10. ** c_exp
            #print("Testing with c=%f" % (c))
            clf = svm.LinearSVC(C=c)
            clf.fit(X_train, y_train)
            y_predicted = clf.predict(X_test)
            # f1 = f1_score(y_test, y_predicted, pos_label=goal_ind)
            p,r,f1,_ = precision_recall_fscore_support(y_test, y_predicted, average='binary', pos_label=goal_ind)
            if f1 > best_f:
                best_f = f1
                best_r = r
                best_p = p
                best_c = c

        print("P/R/F if we tune c=%f to optimize test set F score is %f\t%f\t%f"%  (best_c, best_p, best_r, best_f))



if __name__ == "__main__":
    main(sys.argv[1:])
