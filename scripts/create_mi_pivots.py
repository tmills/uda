#!/usr/bin/env python

import sys
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import mutual_info_classif as mi
import numpy as np
import scipy.sparse
from os.path import dirname, join
from uda_common import read_feature_groups

def main(args):
    if len(args) < 2:
        sys.stderr.write("Error: Two required arguments:  <reduced training data> <0|1 (which domain is source/target)\n")
        sys.exit(-1)

    num_pivots = 1000
    data_file = args[0]
    direction = int(args[1])

    data_dir = dirname(data_file)
    groups_file = join(data_dir, 'reduced-feature-groups.txt')

    ## Find the feature index that tells us what domain we're in:
    group_map = read_feature_groups(groups_file)
    domain_indices = group_map["Domain"]
    if direction == 0:
        source_ind, target_ind = domain_indices
    else:
        target_ind, source_ind = domain_indices

    ## load the data:
    all_X, all_y = load_svmlight_file(data_file)
    num_instances, num_feats = all_X.shape

    source_inds = np.where(all_X[:,source_ind].toarray() != 0)[0]
    target_inds = np.where(all_X[:,target_ind].toarray() != 0)[0]

    mi_label = mi(all_X[source_inds,:], all_y[source_inds])

    y_corpus = np.zeros(num_instances)
    y_corpus[source_inds] = 1
    #X_combined = scipy.sparse.lil_matrix((len(y_corpus), num_feats))
    #X_combined[:num_instances] += X_train
    #X_combined[num_instances:] += X_test

    all_X_minus_domain_feats = all_X.copy()
    all_X_minus_domain_feats[:,source_ind] = 0
    all_X_minus_domain_feats[:,target_ind] = 0
    mi_domains = mi(all_X_minus_domain_feats, y_corpus)

    ## mi is between 0 (no information) and 1 (perfect information)
    ## we want the label informatio to be high (~1)
    ## we want the domain information to be low (1-mi ~ 1)
    ## we want both these things to be true so we use *
    mi_joint = mi_label * (1 - mi_domains)

    ## I want high values, so I reverse the list and sort
    ranked_inds = np.argsort(1 - mi_joint)
    pivots = np.sort(ranked_inds[:num_pivots])

    for pivot in pivots:
        print(pivot)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
