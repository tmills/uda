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
    X_train = X_train.tolil()
    (num_instances, num_feats) = X_train.shape
    print("  Data has %d instances and %d features" % (num_instances, num_feats))

    print("Reading pivot index file into a dictionary and zero-ing out pivot features in train")
    ## zero out pivot features:
    pivots = {}
    f = open(args[2], 'r')
    for line in f:
        line.rstrip()
        pivot = int(line)
        pivots[pivot] = 1
        X_train[:,pivot] = 0
    f.close()

    print("Reading pickled transform matrix file")
    ## Transform space using learned matrix
    matrix_file = open(args[1], 'rb')
    theta = pickle.load(matrix_file)
    matrix_file.close()
    print("  Transformation matrix has dimensions (%d,%d)" % theta.shape)

    print("Transforming training data into SVD space")
    new_space = X_train * theta

    print('Creating "new" feature representation in %s' % (args[3]))
    new_out = open(join(args[3], 'new_model.liblinear'), 'w')
    for row_index in range(new_space.shape[0]):
        new_out.write("%d" % y_train[row_index])
        for col_index in range(new_space.shape[1]):
            new_out.write(" %d:%f" % (col_index+1, new_space[row_index,col_index]))
        new_out.write("\n")
    new_out.close()

    ## pivot features only:
    print("Creating pivot-only feature representation in %s" % (args[3]))
    #pivot_only_out = open(join(args[3], 'pivot-only.liblinear'), 'w')
    X_train, y_train = load_svmlight_file(args[0])
    pivot_X_train = np.matrix(np.zeros(X_train.shape))
    ## zero out all but pivot features
    for pivot in pivots.keys():
        pivot_X_train[:, pivot] += X_train[:, pivot]

    dump_svmlight_file(pivot_X_train, y_train, join(args[3], 'pivot-only.liblinear'))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
