#!/usr/bin/env python

from os.path import join
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.model_selection import cross_validate
from scipy.sparse import coo_matrix, hstack, vstack

from validate_processed_blitzer import read_preprocessed_feature_file, get_data_matrix

def main(args):
    if len(args) < 1:
        sys.stderr.write('Error: Two required arguments: <processed data dir>\n')
        sys.exit(-1)

    doms = ['books', 'dvd', 'electronics', 'kitchen']
    for dom1_ind in range(4):
        word_map = {}
        dom1 = doms[dom1_ind]
        dom1_data, word_map = get_data_matrix(join(args[0], dom1), word_map)
        dom1_y = np.zeros( 2000 )
        # this dataset is balanced to have 1000 instances in every class
        dom1_y[:1000] = 1
        dom1_y[1000:] = 2

        for dom2_ind in range(4):
            if dom1_ind == dom2_ind:
                continue

            dom2 = doms[dom2_ind]
            print("%s is the source and %s is the target" % (dom1, dom2))

            dom2_data, _ = get_data_matrix(join(args[0], dom2), word_map)
            dom2_y = np.zeros( 2000 )
            dom2_y[:1000] = 1 
            dom2_y[1000:] = 2

            clf = SGDClassifier(loss='modified_huber', tol=None, max_iter=50, alpha=0.1)
            clf.fit(dom1_data, dom1_y)
            dom2_data = dom2_data.tolil()[:, :dom1_data.shape[1]]
            score = clf.score(dom2_data, dom2_y)

            lr = LogisticRegression(C=0.1)
            lr.fit(dom1_data, dom1_y)
            lr_score = lr.score(dom2_data, dom2_y)
            lr_preds = lr.predict(dom2_data)

            print("SVM Accuracy score is %f, LR accuracy score is %f" % (score, lr_score))
                
if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)