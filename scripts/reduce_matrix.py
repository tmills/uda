#!/usr/bin/python
from sklearn.datasets import load_svmlight_file
from os import listdir
from os.path import isfile, join, basename
from sklearn.linear_model import SGDClassifier
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import lil_matrix
import numpy as np
import pickle
import sys

## Default value from Blitzer et al. (pos taggin)
#proj_dim = 25
## default value frmo blitzer et al. (sentiment)
proj_dim = 50 

def main(args):
    if len(args) < 2:
        sys.stderr.write("Two required arguments: <full matrix file (input)> <reduced matrix file (output)>\n\n")
        sys.exit(-1)

    weight_file = open(args[0], 'rb')
    weight_matrix = lil_matrix(pickle.load(weight_file))
    sys.stderr.write('Projecting theta into lower dimension\n')
    ## Compute svd to get low-dimensional projection
    [U, s, Vh] = svds(weight_matrix, k=proj_dim)
    ## U is n x n. Take subset of rows to get d x n, then transpose to get n x d
    theta = U[:, 0:proj_dim].transpose()
    ## theta is now an n x d projection from the non-pivot feature space into
    ## the d-dimensional correspondence space.
    theta_out = open(args[1], 'wb')
    pickle.dump(theta, theta_out)
    theta_out.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
