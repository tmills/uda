#!/usr/bin/python
from os.path import join
import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import sys

def main(args):
    if len(args) < 4:
        sys.stderr.write("One required argument: <labeled data> <pickled transform matrix> <pivot index file> <output dir>\n\n")
        sys.exit(-1)

    print("Reading input data from %s" % (args[0]))
    X_train, y_train = load_svmlight_file(args[0])
    nopivot_X_train = X_train.tolil()
    pivot_X_train = np.matrix(np.zeros(X_train.shape))
    (num_instances, num_feats) = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))

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

    print("Creating non-pivot-only feature representation in %s" % (args[3]))
    dump_svmlight_file(nopivot_X_train, y_train, join(args[3], 'nonpivot-only.liblinear'))

    ## pivot features only:
    print("Creating pivot-only feature representation in %s" % (args[3]))
    dump_svmlight_file(pivot_X_train, y_train, join(args[3], 'pivot-only.liblinear'))

    print("Reading pickled transform matrix file")
    ## Transform space using learned matrix
    matrix_file = open(args[1], 'rb')
    theta = pickle.load(matrix_file)
    matrix_file.close()
    print("  Transformation matrix has dimensions (%d,%d)" % theta.shape)

    print("Transforming training data into SVD space")
    new_space = nopivot_X_train * theta

    print('Creating "new" feature representation in %s' % (args[3]))
    dump_svmlight_file(new_space, y_train, join(args[3], 'new.liblinear'))

    print("Creating all+new feature representation in %s" % (args[3]))
    all_plus_new = np.matrix(np.zeros((X_train.shape[0], X_train.shape[1] + new_space.shape[1])))
    all_plus_new[:, :X_train.shape[1]] += X_train
    all_plus_new[:, X_train.shape[1]:] += new_space
    dump_svmlight_file(all_plus_new, y_train, join(args[3], 'all_plus_new.liblinear'))

    print("Creating pivot_plus_new feature representation in %s" % (args[3]))
    pivot_plus_new = np.matrix(np.zeros((pivot_X_train.shape[0], pivot_X_train.shape[1] + new_space.shape[1])))
    pivot_plus_new[:, :pivot_X_train.shape[1]] += pivot_X_train
    pivot_plus_new[:, pivot_X_train.shape[1]:] += new_space
    dump_svmlight_file(pivot_plus_new, y_train, join(args[3], 'pivot_plus_new.liblinear'))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
