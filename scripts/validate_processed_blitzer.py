#!/usr/bin/env python

from os.path import join
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from scipy.sparse import coo_matrix, hstack, vstack


def read_preprocessed_feature_file(fn, word_map={}):
    data = []
    row_inds = []
    col_inds = []
    with open(fn) as f:
        row_num = 0
        for line in f:
            feats = line.rstrip().split(" ")
            for feat in feats:
                if feat.startswith('#label#'):
                    continue
                name,val = feat.split(':')
                if not name in word_map:
                    word_map[name] = len(word_map)
                data.append(int(val))
                row_inds.append(row_num)
                col_inds.append(word_map[name])
            row_num += 1
    coo = coo_matrix( (data, (row_inds, col_inds)) )
    return coo, word_map

def get_data_matrix(fn, word_map = {}):

    pos_insts, word_map = read_preprocessed_feature_file(join(fn, 'positive.review'), word_map)
    neg_insts, word_map = read_preprocessed_feature_file(join(fn, 'negative.review'), word_map)
    # add zeros to extend pos_insts for features i didn't see before
    num_new_feats = neg_insts.shape[1] - pos_insts.shape[1]
    pos_insts_extended = hstack([pos_insts, np.zeros((pos_insts.shape[0], num_new_feats))])
    all_train = vstack([pos_insts_extended, neg_insts])

    return all_train, word_map

def main(args):
    if len(args) < 1:
        sys.stderr.write('Error: One required arguments: <domain dir>\n')
        sys.exit(-1)
    
    all_train,_ = get_data_matrix(args[0])

    all_y = np.zeros( 2000 )
    all_y[:1000] = 1
    all_y[1000:] = 2

    scores = cross_validate(SGDClassifier(loss='modified_huber', tol=None, max_iter=50, alpha=0.1),
                    all_train,
                    all_y,
                    cv=5,
                    scoring='accuracy')
    mean_score = scores['test_score'].mean()
    print("Unmodified in-domain 5-fold CV performance: %f" % (mean_score) )
    
if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)