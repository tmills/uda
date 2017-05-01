#!/usr/bin/env python

from os.path import join
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
import scipy.sparse
import sys
from uda_common import align_test_X_train

def main(args):
    if len(args) < 4:
        sys.stderr.write("Four required arguments: <pivot file> <source data file> <target data file> <output directory>\n")
        sys.exit(-1)

    out_dir = args[3]

    print("Reading in data files")
    X_train, y_train = load_svmlight_file(args[1])
    X_train = X_train.tolil()
    X_test, y_test = load_svmlight_file(args[2])
    X_test = align_test_X_train(X_train, X_test)

    num_instances, num_feats = X_train.shape
    num_test_instances, num_test_feats = X_test.shape

    print("Reading in pivot files and creating pivot labels dictionary")
    ## Read pivots file into dictionary:
    pivots = []
    pivot_labels = {}
    for line in open(args[0], 'r'):
        pivot = int(line.strip())
        pivots.append(pivot)
        pivot_labels[pivot] = np.zeros((num_instances+num_test_instances,1))
        pivot_labels[pivot][:num_instances] += X_train[:,pivot]
        pivot_labels[pivot][num_instances:] += X_test[:,pivot]
        pivot_labels[pivot] = np.round(pivot_labels[pivot])

        X_train[:, pivot] = 0
        X_test[:, pivot] = 0

    print("Creating pivot training data matrix")
    pivot_data = np.zeros((num_instances+num_test_instances, num_feats))
    pivot_data[:num_instances,:] += X_train
    pivot_data[num_instances:,:] += X_test
    pivot_data = scipy.sparse.csr_matrix(pivot_data)

    for pivot in pivots:
        out_file = join(out_dir, 'pivot_%s-training.liblinear' % pivot)
        print("Writing file %s " % out_file)
        dump_svmlight_file(pivot_data, pivot_labels[pivot][:,0], out_file)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
