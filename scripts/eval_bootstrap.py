#!/usr/bin/env python
import numpy as np
import numpy.random
import os
from os.path import join,exists,dirname
from sklearn import svm
import sys

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics import f1_score
from uda_common import evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, read_feature_groups, read_feature_lookup

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

    for direction in [0,1]:
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

        print("Original feature space evaluation (AKA no adaptation, AKA pivot+non-pivot)")
        ## C < 1 => more regularization
        ## C > 1 => more fitting to training
        # C_list = [0.001, 0.01, 0.1, 1.0, 10.0]
        (l1_c, l1_f1) = find_best_c(X_train, y_train, scorer=f1_score, penalty='l1', dual=False, pos_label=goal_ind)
        print("Optimizing l1 with cross-validation gives C=%f and f1=%f" % (l1_c, l1_f1))
        (l2_c, l2_f1) = find_best_c(X_train, y_train, pos_label=goal_ind)
        print("Optimizing l2 with cross-validation gives C=%f and f1=%f" % (l2_c, l2_f1))


        run_oracle_silver_tuning(X_train, y_train, X_test, y_test, goal_ind, l2_c)
        #run_silver_tuning(X_train, y_train, X_test, y_test, goal_ind)
        #run_balanced_bootstrapping(X_train, y_train, X_test, y_test, goal_ind)
        #run_enriched_bootstrapping(X_train, y_train, X_test, y_test, goal_ind)

def run_oracle_silver_tuning(X_train, y_train, X_test, y_test, goal_ind, l2_c):
    print("Tuning regularization using oracle labels from confident/unconfident instances")
    num_train_instances = X_train.shape[0]
    num_feats = X_train.shape[1]
    num_test_instances = X_test.shape[0]

    num_added = 100
    svc = svm.LinearSVC(penalty='l2', C=l2_c)
    svc.fit(X_train, y_train)
    # preds = svc.predict(X_test)  # predict should return 0/1 (maybe experiment with weights?)
    decisions = svc.decision_function(X_test)
    # confidence is signed, so we take the absolute value to get most/least confident
    # decisions in either direction
    confidence_indices = np.argsort(np.abs(decisions))

    for config in ('most_confident', 'least_confident', 'random'):
        if config == 'most_confident':
            target_indices = confidence_indices[-num_added:]
        elif config == 'least_confident':
            target_indices = confidence_indices[:num_added]
        else:
            target_indices = np.random.choice(num_test_instances, num_added, replace=False)

        train_plus_test_X = np.zeros((num_train_instances+num_added, num_feats))
        train_plus_test_X[:num_train_instances, :] += X_train
        train_plus_test_X[num_train_instances:, :] += X_test[target_indices, :]
        train_plus_test_y = np.zeros(num_train_instances+num_added)
        train_plus_test_y[:num_train_instances] += y_train
        train_plus_test_y[num_train_instances:] += y_test[target_indices]
        (gs_l2_c, gs_l2_f1) = find_best_c(train_plus_test_X, train_plus_test_y, pos_label=goal_ind)
        print(" Optimized l2 for gold+silver with %d %s instances added: %f" % (num_added, config, gs_l2_c))
        evaluate_and_print_scores(X_train, y_train, X_test, y_test, goal_ind, gs_l2_c)
        del train_plus_test_X, train_plus_test_y




def run_silver_tuning(X_train, y_train, X_test, y_test, goal_ind):
    print("Tuning regularization parameter on gold+silver:")
    num_train_instances = X_train.shape[0]
    num_feats = X_train.shape[1]
    num_test_instances = X_test.shape[0]

    svc = svm.LinearSVC(penalty='l2', C=l2_c)
    svc.fit(X_train, y_train)
    preds = svc.predict(X_test)  # predict should return 0/1 (maybe experiment with weights?)
    decisions = svc.decision_function(X_test)
    for threshold in (0.0, 1.0):
        sys.stderr.write("Testing silver optimization with confidence threshold %f" % threshold)
        confident_indices = np.where(np.abs(decisions) > threshold)[0]
        num_added = len(confident_indices)

        train_plus_test_X = np.zeros((num_train_instances+num_added, num_feats))
        train_plus_test_X[:num_train_instances, :] += X_train
        train_plus_test_X[num_train_instances:, :] += X_test[confident_indices, :]
        train_plus_test_y = np.zeros(num_train_instances+num_added)
        train_plus_test_y[:num_train_instances] += y_train
        train_plus_test_y[num_train_instances:] += preds[confident_indices]
        (gs_l2_c, gs_l2_f1) = find_best_c(train_plus_test_X, train_plus_test_y, pos_label=goal_ind)
        print(" Optimized l2 for gold+silver: %f" % (gs_l2_c))
        evaluate_and_print_scores(X_train, y_train, X_test, y_test, goal_ind, gs_l2_c)
        del train_plus_test_X, train_plus_test_y


def run_balanced_bootstrapping(X_train, y_train, X_test, y_test, goal_ind):
    print("Balanced bootstrapping method (add equal amounts of true/false examples)")
    num_train_instances = X_train.shape[0]
    num_feats = X_train.shape[1]
    num_test_instances = X_test.shape[0]

    for percentage in [0.01, 0.1, 0.25]:
        svc = svm.LinearSVC()
        svc.fit(X_train, y_train)
        preds = svc.decision_function(X_test)
        added_X = []
        added_y = []
        for i in range(int(percentage * num_test_instances)):
            if i % 2 == 0:
                highest_ind = preds.argmax()
                if preds[highest_ind] <= 0:
                    break
            else:
                highest_ind = preds.argmin()
                if preds[highest_ind] >= 0:
                    break

            added_X.append(X_test[highest_ind,:].toarray()[0])
            added_y.append(1 if preds[highest_ind] < 0 else 2)
            preds[highest_ind] = 0

        print("Added %d instances from target dataset" % (len(added_y)))
        train_plus_bootstrap_X = np.zeros((num_train_instances + len(added_y), num_feats))
        train_plus_bootstrap_X[:num_train_instances, :] += X_train
        train_plus_bootstrap_X[num_train_instances:, :] += np.array(added_X)
        train_plus_bootstrap_y = np.zeros(num_train_instances + len(added_y))
        train_plus_bootstrap_y[:num_train_instances] += y_train
        train_plus_bootstrap_y[num_train_instances:] += np.array(added_y)
        (l2_c, l2_f1) = find_best_c(train_plus_bootstrap_X, train_plus_bootstrap_y, pos_label=goal_ind)
        evaluate_and_print_scores(train_plus_bootstrap_X, train_plus_bootstrap_y, X_test, y_test, goal_ind, l2_c)
        del train_plus_bootstrap_y, train_plus_bootstrap_X

def run_enriched_bootstrapping(X_train, y_train, X_test, y_test, goal_ind):
    print("Enriching bootstrapping method (add minority class examples only)")
    for percentage in [0.01, 0.1, 0.25]:
        svc = svm.LinearSVC()
        svc.fit(X_train, y_train)
        preds = svc.decision_function(X_test)
        added_X = []
        added_y = []
        for i in range(int(percentage * num_test_instances)):
            highest_ind = preds.argmax()
            if preds[highest_ind] <= 0:
                break
            added_X.append(X_test[highest_ind,:].toarray()[0])
            added_y.append(goal_ind)
            preds[highest_ind] = 0

        print("Added %d positive instances from target dataset" % (len(added_y)))
        train_plus_bootstrap_X = np.zeros((num_train_instances + len(added_y), num_feats))
        train_plus_bootstrap_X[:num_train_instances, :] += X_train
        train_plus_bootstrap_X[num_train_instances:, :] += np.array(added_X)
        train_plus_bootstrap_y = np.zeros(num_train_instances + len(added_y))
        train_plus_bootstrap_y[:num_train_instances] += y_train
        train_plus_bootstrap_y[num_train_instances:] += np.array(added_y)
        (l2_c, l2_f1) = find_best_c(train_plus_bootstrap_X, train_plus_bootstrap_y, pos_label=goal_ind)
        evaluate_and_print_scores(train_plus_bootstrap_X, train_plus_bootstrap_y, X_test, y_test, goal_ind, l2_c)
        del train_plus_bootstrap_y, train_plus_bootstrap_X

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
