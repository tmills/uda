#!/usr/bin/python
import numpy as np
import os
from sklearn import svm
import sys

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c

def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <labeled source data> <labeled target data>\n\n")
        sys.exit(-1)

    goal_ind = 2
    (source_file, target_file) = args

    print("Reading input data from %s" % (source_file))
    X_train, y_train = load_svmlight_file(source_file, dtype='float32')
    num_instances, num_feats = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))

    X_test, y_test = load_svmlight_file(target_file)
    X_test = align_test_X_train(X_train, X_test)
    num_test_instances = X_test.shape[0]

    print("Original feature space evaluation (AKA no adaptation, AKA pivot+non-pivot)")
    ## C < 1 => more regularization
    ## C > 1 => more fitting to training
    C_list = [0.001, 0.01, 0.1, 1.0, 10.0]
    (l1_c, l1_f1) = find_best_c(X_train, y_train, goal_ind, penalty='l1', dual=False)
    print("Optimizing l1 with cross-validation gives C=%f and f1=%f" % (l1_c, l1_f1))
    (l2_c, l2_f1) = find_best_c(X_train, y_train, goal_ind)
    print("Optimizing l2 with cross-validation gives C=%f and f1=%f" % (l2_c, l2_f1))

    print("Balanced bootstrapping method (add equal amounts of true/false examples)")
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
        train_plus_bootstrap_X = np.zeros((num_instances + len(added_y), num_feats))
        train_plus_bootstrap_X[:num_instances, :] += X_train
        train_plus_bootstrap_X[num_instances:, :] += np.array(added_X)
        train_plus_bootstrap_y = np.zeros(num_instances + len(added_y))
        train_plus_bootstrap_y[:num_instances] += y_train
        train_plus_bootstrap_y[num_instances:] += np.array(added_y)
        (l2_c, l2_f1) = find_best_c(train_plus_bootstrap_X, train_plus_bootstrap_y, goal_ind)
        evaluate_and_print_scores(train_plus_bootstrap_X, train_plus_bootstrap_y, X_test, y_test, goal_ind, l2_c)
        del train_plus_bootstrap_y, train_plus_bootstrap_X

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
        train_plus_bootstrap_X = np.zeros((num_instances + len(added_y), num_feats))
        train_plus_bootstrap_X[:num_instances, :] += X_train
        train_plus_bootstrap_X[num_instances:, :] += np.array(added_X)
        train_plus_bootstrap_y = np.zeros(num_instances + len(added_y))
        train_plus_bootstrap_y[:num_instances] += y_train
        train_plus_bootstrap_y[num_instances:] += np.array(added_y)
        (l2_c, l2_f1) = find_best_c(train_plus_bootstrap_X, train_plus_bootstrap_y, goal_ind)
        evaluate_and_print_scores(train_plus_bootstrap_X, train_plus_bootstrap_y, X_test, y_test, goal_ind, l2_c)
        del train_plus_bootstrap_y, train_plus_bootstrap_X

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
