#!/usr/bin/env python
import numpy as np
import pickle
import scipy.sparse
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
    print("Accuracy is %f, p/r/f1 score is %f %f %f\n" % (acc, p, r, f1))

def main(args):
    if len(args) < 3:
        sys.stderr.write("Required arguments: <training file> <test file> <model directory>\n")
        sys.exit(-1)

    X_train, y_train = load_svmlight_file(args[0])
    num_instances, num_feats = X_train.shape

    ## Read test and trim feature space to fit training if necessary
    X_test, y_test = load_svmlight_file(args[1])
    num_test_instances, num_test_feats = X_test.shape
    if num_test_feats < num_feats:
        ## Expand X_test
        #print("Not sure I need to do anything here.")
        X_test_array = X_test.toarray()
        X_test = scipy.sparse.csr_matrix(np.append(X_test_array, np.zeros((num_test_instances, num_feats-num_test_feats)), axis=1))
    elif num_test_feats > num_feats:
        ## Truncate X_test
        X_test = X_test[:,:num_feats]

    ## Evaluation 1: No domain adaptation:
    ## FIXME -- magic number 2 is the class of interest : Negated, need to
    ## read this from our outcome lookup file.
    print("Original feature space evaluation (AKA no adaptation, AKA pivot+non-pivot)")
    evaluate_and_print_scores(X_train, y_train, X_test, y_test, 2)

    ## Evaluation 2: "New" feature set: Mapping from non-pivot features into SVD pivot space:
    theta_file = open(join(args[2], 'theta_svd.pkl'), 'rb')
    theta = pickle.load(theta_file)
    num_new_feats = theta.shape[1]
    theta_file.close()

    ## Evaluation 3: Only pivot features used to train classifiers
    print("Pivot-only feature space evaluation")
    pivot_X_train, pivot_y_train = load_svmlight_file(join(args[2], 'transformed/pivot-only.liblinear'), n_features=num_feats)
    evaluate_and_print_scores(pivot_X_train, pivot_y_train, X_test, y_test, 2)

    ## Evaluation 4: Only non-pivot features used to train classifiers
    print("Non-pivot only feature space evaluation")
    nopivot_X_train, nopivot_y_train = load_svmlight_file(join(args[2], 'transformed/nonpivot-only.liblinear'), n_features=num_feats)
    evaluate_and_print_scores(nopivot_X_train, nopivot_y_train, X_test, y_test, 2)

    ## Evaluation 5: Only projected ("new") features used to train classifiers
    new_X_train, new_y_train = load_svmlight_file(join(args[2], 'transformed/new.liblinear'))
    new_X_test = X_test * theta
    print("New-only feature space evaluation")
    evaluate_and_print_scores(new_X_train, new_y_train, new_X_test, y_test, 2)

    ## Evaluation 6: All original plus all new features used to train classifiers
    print("All + new feature space evaluation")
    allnew_X_train, allnew_y_train = load_svmlight_file(join(args[2], 'transformed/all_plus_new.liblinear'), n_features=num_feats+num_new_feats)
    all_plus_new_X_test = np.matrix(np.zeros((X_test.shape[0], num_feats+num_new_feats)))
    all_plus_new_X_test[:, :num_feats] += X_test
    all_plus_new_X_test[:, num_feats:] += new_X_test
    evaluate_and_print_scores(allnew_X_train, allnew_y_train, all_plus_new_X_test, y_test, 2)

    ## Evaluation 7: Pivot features plus all new features used to train classifiers
    print("Pivot + new feature space evaluation")
    pivotnew_X_train, pivotnew_y_train = load_svmlight_file(join(args[2], 'transformed/pivot_plus_new.liblinear'), n_features=num_feats+num_new_feats)
    ## Think we can be cute here -- since the model trained on these files will have no weights for non-pivot features, we can just reuse the allnew_X_test
    ## matrix at classification time:
    evaluate_and_print_scores(pivotnew_X_train, pivotnew_y_train, all_plus_new_X_test, y_test, 2)

    ## Evaluation 8: Yu and Jiang method using similarity with random sample of target features
    num_exemplars = 50
    indices = np.sort(np.random.choice(num_test_instances, num_exemplars, replace=False))
    test_exemplars = X_test[indices]
    ## Normalize
    test_exemplars /= test_exemplars.sum(1)
    ## Create a new feature for every train instance that is the similarity with each of these exemplars:
    ## Output matrix is num_train_instances x num_exemplars. add this to end of X_train:
    similarity_features_train = X_train * test_exemplars.transpose()
    similarity_features_test = X_test * test_exemplars.transpose()
    all_plus_sim_X_train = np.matrix(np.zeros((num_instances, num_feats + num_exemplars)))
    all_plus_sim_X_train[:, :num_feats] += X_train
    all_plus_sim_X_train[:, num_feats:] += similarity_features_train
    all_plus_sim_X_test = np.matrix(np.zeros((num_test_instances, num_feats + num_exemplars)))
    all_plus_sim_X_test[:, :num_feats] += X_test
    all_plus_sim_X_test[:,num_feats:] += similarity_features_test
    evaluate_and_print_scores(all_plus_sim_X_train, y_train, all_plus_sim_X_test, y_test, 2)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
