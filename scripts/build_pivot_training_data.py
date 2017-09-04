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
    group_name = join(model_dir, 'feature-groups.txt')
    group_map = read_feature_groups(group_name)

    out_dir = args[2]

    sys.stderr.write("Reading in data files\n")
    all_X, all_y = load_svmlight_file(args[1])

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
        #pivot_labels[pivot][:num_instances] += X_train[:,pivot]
        #pivot_labels[pivot][num_instances:] += X_test[:,pivot]
        #pivot_labels[pivot] = np.round(pivot_labels[pivot])

    sys.stderr.write("Creating pivot matrices for each feature group\n")
    #ind_groups = [None] * num_feats
    for group_key,group_inds in group_map.items():
        group_inds = np.array(group_inds) - 1
        group_X = scipy.sparse.lil_matrix(np.zeros((num_instances, num_feats)))
        group_X += all_X
        group_X[:, group_inds] = 0
        for group_ind in group_inds:
            if group_ind in pivots:
                out_file = join(out_dir, 'pivot_%s-training.liblinear' % group_ind)
                print('Writing file %s ' % out_file)
                sys.stderr.write('.')
                dump_svmlight_file(group_X, pivot_labels[group_ind][:,0], out_file)

    #     for ind in group_map[domain]:
    #         ind_groups[ind] = domain
    # sys.stderr.write("Creating pivot training data matrix\n")
    # pivot_data = np.zeros((num_instances+num_test_instances, num_feats))
    # pivot_data[:num_instances,:] += X_train
    # pivot_data[num_instances:,:] += X_test
    # sys.stderr.write("Converting pivot-aware matrix to sparse format\n")
    # ## csc does 47 in a minute, lil does 39 in a minute., csr does 47 in a minute, coo does 48
    # ## dok does 37
    # pivot_data = scipy.sparse.coo_matrix(pivot_data)
    #
    # for pivot in pivots:
    #     out_file = join(out_dir, 'pivot_%s-training.liblinear' % pivot)
    #     print("Writing file %s " % out_file)
    #     sys.stderr.write('.')
    #     dump_svmlight_file(pivot_data, pivot_labels[pivot][:,0], out_file)

    sys.stderr.write('\n')

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
