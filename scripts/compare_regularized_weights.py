#!/usr/bin/env python
import numpy as np
import numpy.random
import os
from os.path import join,exists,dirname
from sklearn import svm
import sys

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics import f1_score, precision_recall_fscore_support
from uda_common import evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, read_feature_groups, read_feature_lookup
from scipy.stats import rankdata
from sklearn.feature_selection import chi2

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <combined data>\n\n")
        sys.exit(-1)
    
    goal_ind = 2
    data_file = args[0]
    data_dir = dirname(data_file)

    print("Reading input data from %s" % (data_file))
    all_X, all_y = load_svmlight_file(data_file)
    num_total_instances, num_feats = all_X.shape

    domain_map = read_feature_groups(join(data_dir, 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(data_dir, 'reduced-features-lookup.txt'))

    ## First tune l2_c and get weights for that classifier:
    for direction in [0,1]:
        sys.stderr.write("using domain %s as source, %s as target\n"  %
            (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

        train_instance_inds = np.where(all_X[:,domain_inds[direction]].toarray() > 0)[0]
        X_train = all_X[train_instance_inds,:]
        y_train = all_y[train_instance_inds]
        num_train_instances = X_train.shape[0]
        chi, _ = chi2(X_train, y_train)

        test_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
        X_test = all_X[test_instance_inds,:]
        y_test = all_y[test_instance_inds]
        num_test_instances = X_test.shape[0]

        ## First get the weights when C is tuned to maximize cross-validation
        ## performance in the source domain
        (l2_c, l2_f1) = find_best_c(X_train, y_train, pos_label=goal_ind)
        svc = svm.LinearSVC(penalty='l2', C=l2_c)
        svc.fit(X_train, y_train)
        tuned_weights = svc.coef_[0]
        y_predicted = svc.predict(X_test)
        p,r,f1,_ = precision_recall_fscore_support(y_test, y_predicted, average='binary', pos_label=goal_ind)
        print("With C=%f, Precision,recall,F-score on test data is %f\t%f\t%f" % (l2_c,p,r,f1))

        ## Now get the weights when C is tuned by an oracle to maximize
        ## target-domain performance
        best_c = best_f = best_r = best_p = predicted_prevalence = 0
        for c_exp in range(-10, 0):
            c = 2. ** c_exp
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
                predicted_prevalence = len(np.where(y_predicted == goal_ind)[0]) / float(len(y_test))
        svc = svm.LinearSVC(penalty='l2', C=best_c)
        svc.fit(X_train, y_train)
        oracle_weights = svc.coef_[0]
        print("P/R/F if we tune c=%f to optimize test set F score is %f\t%f\t%f with prevalence %f"%  (best_c, best_p, best_r, best_f, predicted_prevalence))

        print("*****************************")
        tuned_norm_weights = np.abs(tuned_weights) / np.abs(tuned_weights).sum()
        oracle_norm_weights = np.abs(oracle_weights) / np.abs(oracle_weights).sum()
        norm_weight_diffs = tuned_norm_weights - oracle_norm_weights
        norm_highest_diffs = np.argsort(norm_weight_diffs)
        for ind in reversed(norm_highest_diffs[-10:]):
            feature_name = feature_map[ind]
            freq = X_train[:,ind].sum()
            diff = tuned_norm_weights[ind] - oracle_norm_weights[ind]
            print(" Feature %s (freq %0.1f, chi2=%0.4f) had normalized weight change %f (%f to %f)" % (feature_name, freq, chi[ind], diff, tuned_norm_weights[ind], oracle_norm_weights[ind]))

        print("*****************************")
        tuned_ranks = rankdata(np.abs(tuned_weights))
        oracle_ranks = rankdata(np.abs(oracle_weights))
        rank_differences = tuned_ranks - oracle_ranks
        ranked_rank_differences = np.argsort(np.abs(rank_differences))
        for ind in ranked_rank_differences[-10:]:
            feature_name = feature_map[ind]
            print(" Feature %s rank %0.1f -> %0.1f with oracle tuning (%f -> %f)" % (feature_name, tuned_ranks[ind], oracle_ranks[ind], tuned_weights[ind], oracle_weights[ind]))

        print("*****************************")
        weight_differences = tuned_weights - oracle_weights
        highest_diff_inds = np.argsort(np.abs(weight_differences))
        for ind in reversed(highest_diff_inds[-10:]):
            feature_name = feature_map[ind]
            print(" Feature %s had value change from %f to %f after oracle regularization" % (feature_name, tuned_weights[ind], oracle_weights[ind]))
        print("")



if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)