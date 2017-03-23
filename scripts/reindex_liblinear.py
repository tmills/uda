#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import sys

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error: At least one required argument: <Filename>*\n")
        sys.exit(-1)

    ## FIXME - write to standard out.
    X_train, y_train = load_svmlight_file(args[0])
    dump_svmlight_file(X_train, y_train, sys.stdout)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
