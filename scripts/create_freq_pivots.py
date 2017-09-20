#!/usr/bin/env python
import sys
from uda_common import read_feature_groups
from sklearn.datasets import load_svmlight_file
import numpy as np

def main(args):
    if len(args) < 2:
        sys.stderr.write("One required argument: <training data> <feature group file> [freq=50]\n")
        sys.exit(-1)

    freq_cutoff = 50 if len(args) <=2 else args[2]
    ## Find the feature index that tells us what domain we're in:
    group_map = read_feature_groups(args[1])
    domain_indices = group_map["Domain"]

    ## load the data:
    all_X, all_y = load_svmlight_file(args[0])
    num_instances, num_feats = all_X.shape

    data_X = []
    ## To start with, the set of valid_inds is all indices
    ## This prevents the zero index as a pivot (probably the intercept)
    valid_inds = set(range(1, num_feats))
    ## Create a subset for each domain:
    for domain_ind in domain_indices:
        inst_inds = np.where(all_X[:,domain_ind].toarray() != 0)[0]
        ## Find all the variables that are sometimes greater than 0
        nz_inds = set(np.where(all_X[inst_inds,:].max(0).toarray() > 0)[1])
        ## Find variables that are never greater than 1
        lo_inds = set(np.where(all_X[inst_inds,:].max(0).toarray() <= 1)[1])
        ## Take the intersection
        range_inds = nz_inds.intersection(lo_inds)

        ## Find those with high frequency
        freq_inds = set(np.where(all_X[inst_inds].sum(0) > freq_cutoff)[1])

        ## Intersect high freq with correct range, then with existing valid ind
        valid_inds = valid_inds.intersection(range_inds.intersection(freq_inds))

    ind_list = list(valid_inds)
    ind_list.sort()
    for i in ind_list:
        print(i)

if __name__ == "__main__":
    main(sys.argv[1:])
