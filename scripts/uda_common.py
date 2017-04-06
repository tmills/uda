#!/usr/bin/env python
import numpy as np
import scipy.sparse
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score

def remove_pivot_columns(matrix, pivots):
    matrix_lil = matrix.tolil()
    for pivot in pivots:
        matrix_lil[:,pivot] = 0.0

    return matrix_lil.tocsr()

def remove_nonpivot_columns(matrix, pivots):
    matrix_return = np.matrix(np.zeros(matrix.shape))

    for pivot in pivots:
        matrix_return[:,pivot] += matrix[:,pivot]

    return scipy.sparse.csr_matrix(matrix_return)

def evaluate_and_print_scores(X_train, y_train, X_test, y_test, score_label, C=1.0, sample_weight=None, penalty='l2', loss='squared_hinge', dual=True):
    svc = svm.LinearSVC(C=C, penalty=penalty, loss=loss, dual=dual)
    svc.fit(X_train, y_train, sample_weight=sample_weight)
    preds = svc.predict(X_test)

    r = recall_score(y_test, preds, pos_label=score_label)
    p = precision_score(y_test, preds, pos_label=score_label)
    f1 = f1_score(y_test, preds, pos_label=score_label)
    acc = svc.score(X_test, y_test)
    print("Gold has %d instances of target class" % (len(np.where(y_test == score_label)[0])))
    print("System predicted %d instances of target class" % (len(np.where(preds == score_label)[0])))
    print("Accuracy is %f, p/r/f1 score is %f %f %f\n" % (acc, p, r, f1))

def read_pivots(pivot_file):
    pivots = {}
    f = open(pivot_file, 'r')
    for line in f:
        line.rstrip()
        pivot = int(line)
        pivots[pivot] = 1
        ## Before we zero out the nopivot, copy it to the pivot
    f.close()

    return pivots

def align_test_X_train(X_train, X_test):
    num_instances, num_feats = X_train.shape
    num_test_instances, num_test_feats = X_test.shape
    if num_test_feats < num_feats:
        ## Expand X_test
        #print("Not sure I need to do anything here.")
        X_test_array = X_test.toarray()
        X_test = scipy.sparse.csr_matrix(np.append(X_test_array, np.zeros((num_test_instances, num_feats-num_test_feats)), axis=1))
    elif num_test_feats > num_feats:
        ## Truncate X_test
        X_test = X_test[:,:num_feats]

    return X_test
