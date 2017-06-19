#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from uda_common import align_test_X_train
import sys
import numpy as np
import scipy.sparse

def main(args):
    if len(args) < 2:
        sys.stderr.write("Error: Requires two input dataset arguments.")
        sys.exit(-1)

    source_X, source_y = load_svmlight_file(args[0])
    num_source_instances, num_features = source_X.shape
    target_X, target_y = load_svmlight_file(args[1])
    target_X = align_test_X_train(source_X, target_X)
    num_target_instances, num_target_features = target_X.shape
    source_fn = args[0][:args[0].rfind('.')]
    target_fn = args[1][:args[1].rfind('.')]
    out_fn = source_fn+"+"+target_fn

    ## Create one dataset with both domains:
    combined_X = scipy.sparse.lil_matrix(np.zeros((num_source_instances+num_target_instances, num_features)))
    combined_X[:num_source_instances,:] += source_X
    combined_X[num_source_instances:,:] += target_X
    ## The domain variable has 4 values: 2 domains x 2 labels.
    color_y = np.zeros(num_source_instances+num_target_instances)
    color_y[:num_source_instances] += source_y
    color_y[num_source_instances:] += (target_y+2)

    reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(combined_X)
    X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(reduced)
    np.save('%s_X.npy' % (out_fn), (X_embedded[:num_source_instances,:], X_embedded[num_source_instances:,:]))
    np.save('%s_color_y.npy' % (out_fn), (color_y[:num_source_instances], color_y[num_source_instances:]))

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
