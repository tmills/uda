#!/usr/bin/env python

from os.path import join, dirname
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
import scipy.sparse
import sys
from uda_common import read_feature_groups

def main(args):
    if len(args) < 3:
        sys.stderr.write("Three required arguments: <pivot file> <data file> <output directory>\n")
        sys.exit(-1)

    pivot_file = args[0]
    model_dir = dirname(pivot_file)
    group_name = join(model_dir, 'reduced-feature-groups.txt')
    group_map = read_feature_groups(group_name)
    domain_inds = group_map['Domain']

    out_dir = args[2]

    sys.stderr.write("Reading in data files\n")
    all_X, all_y = load_svmlight_file(args[1])
    ## Zero out domain-indicator variables (not needed for this step)
    all_X[:,domain_inds[0]] = 0
    all_X[:,domain_inds[1]] = 0
    num_instances, num_feats = all_X.shape

    sys.stderr.write("Reading in pivot files and creating pivot labels dictionary\n")
    ## Read pivots file into dictionary:
    pivots = []
    pivot_labels = {}
    for line in open(pivot_file, 'r'):
        pivot = int(line.strip())
        pivots.append(pivot)
        pivot_labels[pivot] = np.zeros((num_instances,1))
        pivot_labels[pivot] += np.round(all_X[:,pivot]).toarray()

    sys.stderr.write("Creating pivot matrices for each feature group\n")
    #ind_groups = [None] * num_feats
    for group_key,group_inds in group_map.items():
        group_inds = np.array(group_inds)
        group_X = scipy.sparse.lil_matrix(np.zeros((num_instances, num_feats)))
        group_X += all_X
        group_X[:, group_inds] = 0
        group_X[:, pivots] = 0
        for group_ind in group_inds:
            if group_ind in pivots:
                out_file = join(out_dir, 'pivot_%s-training.liblinear' % group_ind)
                print('Writing file %s ' % out_file)
                sys.stderr.write('.')
                dump_svmlight_file(group_X, pivot_labels[group_ind][:,0], out_file)

    sys.stderr.write('\n')

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
