#!/usr/bin/env python
import numpy as np
import pickle
import sys
from os.path import isfile, join, basename
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score, recall_score, precision_score

def evaluate_and_print_scores(X_train, y_train, X_test, y_test, score_label):
    svc = svm.LinearSVC()
    svc.fit(X_train, y_train)
    preds = svc.predict(X_test)

    r = recall_score(y_test, preds, pos_label=score_label)
    p = precision_score(y_test, preds, pos_label=score_label)
    f1 = f1_score(y_test, preds, pos_label=score_label)
    acc = svc.score(X_test, y_test)
    print("Gold has %d instances of target class" % (len(np.where(y_test == score_label)[0])))
    print("System predicted %d instances of target class" % (len(np.where(preds == score_label)[0])))
    print("Accuracy is %f, p/r/f1 score is %f %f %f" % (acc, p, r, f1))

def main(args):
    if len(args) < 3:
        sys.stderr.write("Required arguments: <training file> <test file> <model directory>")
        sys.exit(-1)

    X_train, y_train = load_svmlight_file(args[0])
    num_instances, num_feats = X_train.shape

    ## Read test and trim feature space to fit training if necessary
    X_test, y_test = load_svmlight_file(args[1], n_features=num_feats)

    ## Evaluation 1: No domain adaptation:
    ## FIXME -- magic number 2 is the class of interest : Negated, need to
    ## read this from our outcome lookup file.
    print("Original feature space evaluation (no adaptation)")
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, 2)

    ## Evaluation 2: "New" feature set: Mapping from non-pivot features into SVD pivot space:
    theta_file = open(join(args[2], 'theta.pkl'), 'rb')
    theta = pickle.load(theta_file)
    theta_file.close()

    new_X_train, new_y_train = load_svmlight_file(join(args[2], 'transformed/new_model.liblinear'))
    new_X_test = X_test * theta
    print("New-only feature space evaluation")
    evaluate_and_print_scores(new_X_train, new_y_train, new_X_test, y_test, 2)

    print("Pivot-only feature space evaluation")
    pivot_X_train, pivot_y_train = load_svmlight_file(join(args[2], 'transformed/pivot-only.liblinear'), n_features=num_feats)
    evaluate_and_print_scores(pivot_X_train, pivot_y_train, X_test, y_test, 2)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
