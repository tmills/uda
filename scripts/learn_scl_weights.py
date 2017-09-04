#!/usr/bin/python
from sklearn.datasets import load_svmlight_file
from os import listdir
from os.path import isfile, join, basename
from sklearn.linear_model import SGDClassifier
from scipy.linalg import svd
import numpy as np
import pickle
import sys


def main(args):
    if len(args) < 2:
        sys.stderr.write("Two required arguments: <data directory> <output file>\n\n")
        sys.exit(-1)

    data_dir = args[0]
    files = [join(data_dir,f) for f in listdir(data_dir) if f.endswith("liblinear")]
    weight_matrix = None

    for ind,f in enumerate(files):
        sys.stderr.write("Loading file %s for classification\n" % (f))
        X_train, y_train = load_svmlight_file(f)
        ## Weight matrix is supposed to be n x p, n non-pivot features by p pivot features
        ## Here we just zeroed out all the pivot features in the pre-process, so we
        ## will actually have m x p but with <=n non-zero features.
        if weight_matrix is None:
            num_feats = X_train.shape[1]
            weight_matrix = np.zeros((num_feats, len(files)))
        clf = SGDClassifier(loss="modified_huber", penalty='none', fit_intercept=False)
        clf.fit(X_train, y_train)
        coefs_out = open(join(data_dir, basename(f).replace('liblinear','model') ), 'wb')
        pickle.dump(clf, coefs_out)
        coefs_out.close()

        weight_matrix[:,ind] = clf.coef_

    sys.stderr.write('Writing full theta matrix\n')
    full_out = open(join(data_dir, args[1]), 'wb')
    pickle.dump(weight_matrix, full_out)
    full_out.close()

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
