#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from uda_common import evaluate_and_print_scores, read_feature_groups, read_feature_lookup, find_best_c
import numpy as np
import scipy.sparse
import sys
from os.path import join,exists,dirname

def main(args):
    if len(args) < 1:
        sys.stderr.write("Two required arguments: <combined data file>\n")
        sys.exit(-1)

    data_file = args[0]
    data_dir = dirname(data_file)
    goal_ind = 2

    print("Reading input data from %s" % (data_file))
    all_X, all_y = load_svmlight_file(data_file)
    num_total_instances, num_feats = all_X.shape

    domain_map = read_feature_groups(join(data_dir, 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(data_dir, 'reduced-features-lookup.txt'))

    print("Yu and Jiang method (50 similarity features)")
    for direction in [0,]:
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

        num_exemplars = 100
        indices = np.sort(np.random.choice(num_test_instances, num_exemplars, replace=False))
        test_exemplars = X_test[indices]
        ## Normalize
        test_exemplars /= test_exemplars.sum(1)
        ## Create a new feature for every train instance that is the similarity with each of these exemplars:
        ## Output matrix is num_train_instances x num_exemplars. add this to end of X_train:
        similarity_features_train = X_train * test_exemplars.transpose()
        similarity_features_test = X_test * test_exemplars.transpose()
        all_plus_sim_X_train = np.matrix(np.zeros((num_train_instances, num_feats + num_exemplars)))
        all_plus_sim_X_train[:, :num_feats] += X_train
        all_plus_sim_X_train[:, num_feats:] += similarity_features_train
        all_plus_sim_X_test = np.matrix(np.zeros((num_test_instances, num_feats + num_exemplars)))
        all_plus_sim_X_test[:, :num_feats] += X_test
        all_plus_sim_X_test[:,num_feats:] += similarity_features_test
        l2_c, _ = find_best_c(all_plus_sim_X_train, y_train, pos_label=goal_ind)
        evaluate_and_print_scores(all_plus_sim_X_train, y_train, all_plus_sim_X_test, y_test, goal_ind, l2_c)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
