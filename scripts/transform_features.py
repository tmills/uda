#!/usr/bin/python
from os.path import join,exists
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import os
import sys

def main(args):
    if len(args) < 3:
        sys.stderr.write("One required argument: <labeled data> <matrix directory> <pivot index file>\n\n")
        sys.exit(-1)

    print("Reading input data from %s" % (args[0]))
    X_train, y_train = load_svmlight_file(args[0])
    nopivot_X_train = X_train.tolil()
    pivot_X_train = np.matrix(np.zeros(X_train.shape))
    (num_instances, num_feats) = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))

    out_dir = join(args[1], 'transformed')
    if not exists(out_dir):
        sys.stderr.write("Creating non-existent output directory.\n")
        os.makedirs(out_dir)

    print("Reading pivot index file into a dictionary and creating pivot-only and nopivot matrices")
    ## zero out pivot features:
    pivots = {}
    f = open(args[2], 'r')
    for line in f:
        line.rstrip()
        pivot = int(line)
        pivots[pivot] = 1
        ## Before we zero out the nopivot, copy it to the pivot
        pivot_X_train[:,pivot] += nopivot_X_train[:,pivot]
        nopivot_X_train[:,pivot] = 0
    f.close()

    print("Creating non-pivot-only feature representation in %s" % (out_dir))
    dump_svmlight_file(nopivot_X_train, y_train, join(out_dir, 'nonpivot-only.liblinear'))

    ## pivot features only:
    print("Creating pivot-only feature representation in %s" % (out_dir))
    dump_svmlight_file(pivot_X_train, y_train, join(out_dir, 'pivot-only.liblinear'))

    print("Reading pickled svd transform matrix file")
    ## Transform space using learned matrix
    matrix_file = open(join(args[1], 'theta_svd.pkl'), 'rb')
    theta = pickle.load(matrix_file)
    matrix_file.close()
    print("  Transformation matrix has dimensions (%d,%d)" % theta.shape)

    print("Transforming training data into SVD space")
    new_space = nopivot_X_train * theta

    print('Creating "new" feature representation in %s' % (out_dir))
    dump_svmlight_file(new_space, y_train, join(out_dir, 'new.liblinear'))

    print("Reading pickled raw transform matrix file")
    with open(join(args[1], 'theta_full.pkl'), 'rb') as f:
        theta_full = pickle.load(f)
    pivot_space = nopivot_X_train * theta_full
    dump_svmlight_file(pivot_space, y_train, join(out_dir, 'pivot_pred.liblinear'))

    print("Creating all+new feature representation in %s" % (out_dir))
    all_plus_new = np.matrix(np.zeros((X_train.shape[0], X_train.shape[1] + new_space.shape[1])))
    all_plus_new[:, :X_train.shape[1]] += X_train
    all_plus_new[:, X_train.shape[1]:] += new_space
    dump_svmlight_file(all_plus_new, y_train, join(out_dir, 'all_plus_new.liblinear'))

    print("Creating pivot_plus_new feature representation in %s" % (out_dir))
    pivot_plus_new = np.matrix(np.zeros((pivot_X_train.shape[0], pivot_X_train.shape[1] + new_space.shape[1])))
    pivot_plus_new[:, :pivot_X_train.shape[1]] += pivot_X_train
    pivot_plus_new[:, pivot_X_train.shape[1]:] += new_space
    dump_svmlight_file(pivot_plus_new, y_train, join(out_dir, 'pivot_plus_new.liblinear'))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
