#!/usr/bin/env python
import numpy as np
import scipy.sparse
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score

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

def evaluate_and_print_scores(X_train, y_train, X_test, y_test, score_label, C, sample_weight=None, penalty='l2', loss='squared_hinge', dual=True):
    preds = get_preds(X_train, y_train, X_test, C, sample_weight=None, penalty='l2', loss='squared_hinge', dual=True)
    r = recall_score(y_test, preds, pos_label=score_label)
    p = precision_score(y_test, preds, pos_label=score_label)
    f1 = f1_score(y_test, preds, pos_label=score_label)
    acc = accuracy_score(y_test, preds)
    print("Gold has %d instances of target class" % (len(np.where(y_test == score_label)[0])))
    print("System predicted %d instances of target class" % (len(np.where(preds == score_label)[0])))
    print("Accuracy is %f, p/r/f1 score is %f %f %f\n" % (acc, p, r, f1))

def get_preds(X_train, y_train, X_test, C=1.0, sample_weight=None, penalty='l2', loss='squared_hinge', dual=True):
    svc = svm.LinearSVC(C=C, penalty=penalty, loss=loss, dual=dual)
    svc.fit(X_train, y_train, sample_weight=sample_weight)
    preds = svc.predict(X_test)
    return preds

def get_f1(X_train, y_train, X_test, y_test, score_label, C=1.0, sample_weight=None, penalty='l2', loss='squared_hinge', dual=True):
    preds = get_preds(X_train, y_train, X_test, C=C, sample_weight=None, penalty='l2', loss='squared_hinge', dual=True)
    f1 = f1_score(y_test, preds, pos_label=score_label)
    return f1

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

def find_best_c(X_train, y_train, goal_ind, C_list = [0.01, 0.1, 1.0, 10.0], penalty='l2', dual=True):
    scorer = make_scorer(f1_score, pos_label=goal_ind)
    best_f1 = 0
    best_c = 0
    for C in C_list:
        f1 = np.average(cross_val_score(svm.LinearSVC(C=C, penalty=penalty, dual=dual), X_train, y_train, scoring=scorer))
        if f1 > best_f1:
            best_f1 = f1
            best_c = C

    return best_c, best_f1
