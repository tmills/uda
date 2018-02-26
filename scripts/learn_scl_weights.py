#!/usr/bin/python
from sklearn.datasets import load_svmlight_file
from os import listdir
from os.path import isfile, join, basename, dirname
# from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from scipy.linalg import svd
import numpy as np
import pickle
import sys
from uda_common import read_pivots


def main(args):
    if len(args) < 2:
        sys.stderr.write("Two required arguments: <data directory> <output file>\n\n")
        sys.exit(-1)

    data_dir = args[0]
    short_dir = basename(data_dir)
    ## data dir is pivot_name_pivots_done so we need to chop off the end:
    pivot_name = short_dir[:-11]
    pivot_logfile = join(dirname(data_dir), pivot_name + '_pivots_done.txt')
    sys.stderr.write("Reading input file names from %s\n" % pivot_logfile)
    files = []
    f = open(pivot_logfile, 'r')
    for line in f:
        line = line.rstrip()
        files.append(line[13:])
    f.close()
    
    #files = [join(data_dir,f) for f in listdir(data_dir) if f.endswith("liblinear")]
    weight_matrix = None

    for ind,f in enumerate(files):
        sys.stderr.write("Loading file %s for classification\n" % (f))
        X_train, y_train = load_svmlight_file(f)
        ## Weight matrix is supposed to be n x p, n non-pivot features by p pivot features
        ## Here we just zeroed out all the pivot features in the pre-process, so we
        ## will actually have m x p but with <=n non-zero features.
        if weight_matrix is None:
            num_feats = X_train.shape[1]
            weight_matrix = np.zeros((num_feats, len(files)), dtype=np.float16)
        # clf = SGDClassifier(loss="modified_huber", penalty='none', fit_intercept=False, random_state=718)
        clf = LinearSVC(fit_intercept=False)
        clf.fit(X_train, y_train)
        coefs_out = open(join(data_dir, basename(f).replace('liblinear','model') ), 'wb')
        pickle.dump(clf, coefs_out)
        coefs_out.close()

        weight_matrix[:,ind] = clf.coef_

    sys.stderr.write('Writing full theta matrix\n')
    full_out = open(args[1], 'wb')
    pickle.dump(weight_matrix, full_out)
    full_out.close()

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
