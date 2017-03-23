#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import sys

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error: At least one required argument: <Filename>*\n")
        sys.exit(-1)

    for arg in args:
        X_train, y_train = load_svmlight_file(arg)
        dump_svmlight_file(X_train, y_train, arg)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
