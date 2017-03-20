#!/usr/bin/python
from sklearn.datasets import load_svmlight_file
from os import listdir
from os.path import isfile, join, basename
from sklearn.linear_model import SGDClassifier
from scipy.linalg import svd
import numpy as np
import pickle
import sys

## Default value from Blitzer et al.
proj_dim = 25

def main(args):
    if len(args) < 1:
        sys.stderr.write("One required argument: <data directory>\n\n")
        sys.exit(-1)

    data_dir = args[0]
    files = [join(data_dir,f) for f in listdir(data_dir) if f.endswith("liblinear")]
    weight_matrix = None

    for ind,f in enumerate(files):
        X_train, y_train = load_svmlight_file(f)
        ## Weight matrix is supposed to be n x p, n non-pivot features by p pivot features
        ## Here we just zeroed out all the pivot features in the pre-process, so we
        ## will actually have m x p but with <=n non-zero features.
        if weight_matrix is None:
            num_feats = X_train.shape[1]
            weight_matrix = np.zeros((num_feats, len(files)))
        print("Training classifier for pivot file %s with dimensions %s" % (f, str(weight_matrix.shape)))
        clf = SGDClassifier(loss="modified_huber", penalty='none', fit_intercept=False)
        clf.fit(X_train, y_train)
        coefs_out = open(join(data_dir, basename(f).replace('liblinear','model') ), 'wb')
        pickle.dump(clf, coefs_out)
        coefs_out.close()

        weight_matrix[:,ind] = clf.coef_

    full_out = open(join(data_dir, 'theta_full.pkl'), 'wb')
    pickle.dump(weight_matrix, full_out)
    full_out.close()

    ## Compute svd to get low-dimensional projection
    [U, s, Vh] = svd(weight_matrix, full_matrices=True)
    ## U is n x n. Take subset of rows to get d x n, then transpose to get n x d
    theta = U[0:proj_dim, :].transpose()
    ## theta is now an n x d projection from the non-pivot feature space into
    ## the d-dimensional correspondence space.
    theta_out = open(join(data_dir, 'theta_svd.pkl'), 'wb')
    pickle.dump(theta, theta_out)
    theta_out.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
