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

    # Blitzer doesn't say how many pivots to use; ziser tries 100-500
    num_pivots = 100
    data_file = args[0]
    direction = int(args[1])
    doc_freq = 5  # blitzer uses 5, ziser uses 10?

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

    source_X = all_X[source_inds,:]
    target_X = all_X[target_inds,:]

    ## map all features to 1 so that sum gives us a document frequency
    source_docfreq_X = scipy.sparse.lil_matrix((source_X > 0).astype(int))
    target_docfreq_X = scipy.sparse.lil_matrix((target_X > 0).astype(int))
    
    docfreq_X = scipy.sparse.lil_matrix((all_X > 0).astype(int))

    freq_mask = np.asarray( ((source_docfreq_X.sum(0) > doc_freq) & (target_docfreq_X.sum(0) > doc_freq)).astype('int') )[0]

    mi_label = mi(source_X, all_y[source_inds])

    ## mi is between 0 (no information) and 1 (perfect information)
    mi_joint = mi_label * freq_mask

    ## I want high values, so I reverse the list and sort
    ranked_inds = np.argsort(1 - mi_joint)
    pivots = np.sort(ranked_inds[:num_pivots])

    for pivot in pivots:
        print(pivot)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
