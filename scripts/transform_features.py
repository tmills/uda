#!/usr/bin/python
from os.path import join,exists
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import remove_pivot_columns, remove_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c
import os
import scipy.sparse
from scipy.sparse import lil_matrix
import sys
from sklearn import svm
from sklearn.feature_selection import chi2

def main(args):
    if len(args) < 4:
        sys.stderr.write("Required argument(s): <labeled source data> <labeled target data> <matrix directory> <pivot index file>\n\n")
        sys.exit(-1)

    goal_ind = 2
    (source_file, target_file, matrix_dir, pivot_file) = args

    print("Reading pivot index file into a dictionary and creating pivot-only and nopivot matrices")
    pivots = read_pivots(pivot_file)
    num_pivots = len(pivots)

    print("Reading input data from %s" % (source_file))
    X_train, y_train = load_svmlight_file(source_file, dtype='float32')
    num_instances, num_feats = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))
    nopivot_X_train = remove_pivot_columns(X_train, pivots)
    pivot_X_train = remove_nonpivot_columns(X_train, pivots)

    X_test, y_test = load_svmlight_file(target_file)
    X_test = align_test_X_train(X_train, X_test)
    num_test_instances = X_test.shape[0]

    nopivot_X_test = remove_pivot_columns(X_test, pivots)
    pivot_X_test = remove_nonpivot_columns(X_test, pivots)

    print("Original feature space evaluation (AKA no adaptation, AKA pivot+non-pivot)")
    ## C < 1 => more regularization
    ## C > 1 => more fitting to training
    C_list = [0.001, 0.01, 0.1, 1.0, 10.0]
    (l1_c, l1_f1) = find_best_c(X_train, y_train, goal_ind, penalty='l1', dual=False)
    print("Optimizing l1 with cross-validation gives C=%f and f1=%f" % (l1_c, l1_f1))
    (l2_c, l2_f1) = find_best_c(X_train, y_train, goal_ind)
    print("Optimizing l2 with cross-validation gives C=%f and f1=%f" % (l2_c, l2_f1))

    print("Evaluating with l1=%f regularization optimized for source:" % l1_c)
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, goal_ind, C=l1_c, penalty='l1', dual=False)

    print("Evaluating l1 at c=%f" % (l1_c/10))
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, goal_ind, C=l1_c/10, penalty='l1', dual=False)

    print("Evaluating with l2=%f regularization optimized for source:" % l2_c)
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, goal_ind, C=l2_c)

    print("Evaluating l2 at c=%f" % (l2_c/10))
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, goal_ind, C=l2_c/10)

    print("Feature selection in source space")
    ## Find features useful for the primary task:
    (chi2_task, pval_task) = chi2(X_train, y_train)
    task_feats_inds = np.where(pval_task < 0.05)[0]
    X_train_featsel = remove_nonpivot_columns(X_train, task_feats_inds)
    (l2_c, l2_f1) = find_best_c(X_train_featsel, y_train, goal_ind)
    X_test_featsel = remove_nonpivot_columns(X_test, task_feats_inds)
    evaluate_and_print_scores(X_train_featsel, y_train, X_test_featsel, y_test, goal_ind, C=l2_c)
    #del X_train_featsel, X_test_featsel

    print("Remove features useful for telling domains apart")
    ## Create dataset for training a classifier to distinguish between domains:
    X_all = np.zeros((num_instances+num_test_instances, num_feats))
    X_all[:num_instances,:] += X_train
    X_all[num_instances:,:] += X_test
    y_dataset_discrim = np.zeros(num_instances+num_test_instances)
    y_dataset_discrim[:num_instances] = 1
    svc = svm.LinearSVC()
    svc.fit(X_all, y_dataset_discrim)
    train_error = 1 - svc.score(X_all, y_dataset_discrim)
    print("Train error in trying to differentiate these datasets is %f" % train_error)

    (chi2_dd, pval_dd) = chi2(X_all, y_dataset_discrim)
    dd_feats_inds = np.where(pval_dd > 0.05)[0]
    X_train_domains = remove_nonpivot_columns(X_train, dd_feats_inds)
    X_test_domains = remove_nonpivot_columns(X_test, dd_feats_inds)
    (l2_c, l2_f1) = find_best_c(X_train_domains, y_train, goal_ind)
    evaluate_and_print_scores(X_train_domains, y_train, X_test_domains, y_test, goal_ind, C=l2_c)
    del X_train_domains, X_test_domains

    print("Ben-david logic with feature selection intersection")
    intersect_inds = np.intersect1d(task_feats_inds, dd_feats_inds)
    X_train_intersect = remove_nonpivot_columns(X_train, intersect_inds)
    X_test_intersect = remove_nonpivot_columns(X_test, intersect_inds)
    (l2_c, l2_f1) = find_best_c(X_train_intersect, y_train, goal_ind)
    evaluate_and_print_scores(X_train_intersect, y_train, X_test_intersect, y_test, goal_ind, C=l2_c)
    del X_train_intersect, X_test_intersect

    ## FIXME -- don't add all at once?
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

    with open(join(matrix_dir, 'theta_svd.pkl'), 'rb') as theta_file:
        theta = pickle.load(theta_file)
        num_new_feats = theta.shape[1]

    with open(join(matrix_dir, 'theta_full.pkl'), 'rb') as theta_file:
        theta_full = pickle.load(theta_file)
        num_pivots = theta_full.shape[1]

    #print("Pivot-only feature space evaluation")
    #evaluate_and_print_scores(pivot_X_train, y_train, pivot_X_test, y_test, goal_ind)

    #print("Non-pivot only feature space evaluation")
    #evaluate_and_print_scores(nopivot_X_train, y_train, nopivot_X_test, y_test, goal_ind)

    #print("New-only feature space evaluation (svd)")
    new_X_train = nopivot_X_train * theta
    new_X_test = nopivot_X_test * theta
    #evaluate_and_print_scores(new_X_train, y_train, new_X_test, y_test, goal_ind)


    print("All + new feature space evaluation")
    all_plus_new_train = lil_matrix((X_train.shape[0], num_feats + num_new_feats), dtype='float32')
    all_plus_new_train[:, :num_feats] += X_train
    all_plus_new_train[:, num_feats:] += new_X_train
    all_plus_new_test = lil_matrix((X_test.shape[0], num_feats + num_new_feats), dtype='float32')
    all_plus_new_test[:, :num_feats] += X_test
    all_plus_new_test[:, num_feats:] += new_X_test
    (l2_c, l2_f1) = find_best_c(all_plus_new_train, y_train, goal_ind)
    evaluate_and_print_scores(all_plus_new_train, y_train, all_plus_new_test, y_test, goal_ind, l2_c)
    del all_plus_new_train, all_plus_new_test

    print("Pivot + new feature space evaluation")
    pivot_plus_new_train = lil_matrix((X_train.shape[0], num_feats + num_new_feats), dtype='float32')
    pivot_plus_new_train[:, :num_feats] += pivot_X_train
    pivot_plus_new_train[:, num_feats:] += new_X_train
    pivot_plus_new_test = lil_matrix((X_test.shape[0], num_feats + num_new_feats), dtype='float32')
    pivot_plus_new_test[:, :num_feats] += pivot_X_test
    pivot_plus_new_test[:, num_feats:] += new_X_test
    (l2_c, l2_f1)=  find_best_c(pivot_plus_new_train, y_train, goal_ind)
    evaluate_and_print_scores(pivot_plus_new_train, y_train, pivot_plus_new_test, y_test, goal_ind, l2_c)
    del pivot_plus_new_train, pivot_plus_new_test

    #print("New-only features space evaluation (no svd)")
    #pivotpred_X_train = nopivot_X_train * theta_full
    #pivotpred_X_test = nopivot_X_test * theta_full
    #evaluate_and_print_scores(pivotpred_X_train, y_train, pivotpred_X_test, y_test, goal_ind)

    #print("All + no-svd pivot feature space")
    # all_plus_pivotpred_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_pivots)))
    # all_plus_pivotpred_train[:, :num_feats] += X_train
    # all_plus_pivotpred_train[:, num_feats:] += pivotpred_X_train
    # all_plus_pivotpred_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_pivots)))
    # all_plus_pivotpred_test[:, :num_feats] += X_test
    # all_plus_pivotpred_test[:, num_feats:] += pivotpred_X_test
    # evaluate_and_print_scores(all_plus_pivotpred_train, y_train, all_plus_pivotpred_test, y_test, goal_ind)
    # del all_plus_pivotpred_train, all_plus_pivotpred_test

    # print("Pivot + pivot prediction space")
    # pivot_plus_pivot_pred_train = np.matrix(np.zeros((X_train.shape[0], num_feats + num_pivots)))
    # pivot_plus_pivot_pred_train[:,:num_feats] += pivot_X_train
    # pivot_plus_pivot_pred_train[:, num_feats:] += pivotpred_X_train
    # pivot_plus_pivot_pred_test = np.matrix(np.zeros((X_test.shape[0], num_feats + num_pivots)))
    # pivot_plus_pivot_pred_test[:, :num_feats] += pivot_X_test
    # pivot_plus_pivot_pred_test[:, num_feats:] += pivotpred_X_test
    # evaluate_and_print_scores(pivot_plus_pivot_pred_train, y_train, pivot_plus_pivot_pred_test, y_test, goal_ind)
    # del pivot_plus_pivot_pred_train, pivot_plus_pivot_pred_test

    print("Original space minus missing target features")
    ## since X_test is a matrix a slice is a matrix and need to get the 2d array and then grab the 0th row to get a 1d array.
    column_sums = abs(X_test).sum(0).A[0,:]
    assert len(column_sums) == X_test.shape[1]
    zero_columns = np.where(column_sums == 0)[0]
    nosrconly_feats_train = lil_matrix(X_train)
    nosrconly_feats_train[:, zero_columns] = 0
    (l2_c, l2_f1) = find_best_c(nosrconly_feats_train, y_train, goal_ind)
    evaluate_and_print_scores(nosrconly_feats_train, y_train, X_test, y_test, goal_ind, l2_c)
    del nosrconly_feats_train

    ## TODO Fix this with feature selection?
    #print("Sun et al. 15 UFEDA")
    #sample_size = 100
    #train_index = min(sample_size, X_train.shape[0])
    #test_index = min(sample_size, X_test.shape[0])
    #cov_train = np.cov(X_train[:train_index,:].toarray(), rowvar=False) + np.eye(X_train.shape[1])
    #cov_test = np.cov(X_test[:test_index,:].toarray(), rowvar=False) + np.eye(X_test.shape[1])
    #whitened_train = X_train * cov_train**-0.5
    #recolored_train = whitened_train * cov_test.transpose()**0.5
    #evaluate_and_print_scores(recolored_train, y_train, X_test, y_test, 2)

    print("Yu and Jiang method (50 similarity features)")
    num_exemplars = 50
    indices = np.sort(np.random.choice(num_test_instances, num_exemplars, replace=False))
    test_exemplars = X_test[indices]
    ## Normalize
    test_exemplars /= test_exemplars.sum(1)
    ## Create a new feature for every train instance that is the similarity with each of these exemplars:
    ## Output matrix is num_train_instances x num_exemplars. add this to end of X_train:
    similarity_features_train = X_train * test_exemplars.transpose()
    similarity_features_test = X_test * test_exemplars.transpose()
    all_plus_sim_X_train = lil_matrix((num_instances, num_feats + num_exemplars), dtype='float32')
    all_plus_sim_X_train[:, :num_feats] += X_train
    all_plus_sim_X_train[:, num_feats:] += similarity_features_train
    all_plus_sim_X_test = lil_matrix((num_test_instances, num_feats + num_exemplars), dtype='float32')
    all_plus_sim_X_test[:, :num_feats] += X_test
    all_plus_sim_X_test[:,num_feats:] += similarity_features_test
    (l2_c, l2_f1) = find_best_c(all_plus_sim_X_train, y_train, goal_ind)
    evaluate_and_print_scores(all_plus_sim_X_train, y_train, all_plus_sim_X_test, y_test, goal_ind, l2_c)
    del all_plus_sim_X_train, all_plus_sim_X_test



if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
