#!/usr/bin/env python
import sys
import os
from os.path import join, basename, dirname
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from scipy.sparse import lil_matrix, hstack, vstack
from sklearn.datasets import dump_svmlight_file

from preprocess_blitzer_raw import get_domain_feature
from validate_processed_blitzer import get_data_matrix

def main(args):
    if len(args) < 3:
        sys.stderr.write('Error: 3 required arguments: <domain 1> <domain 2> <output dir>\n')
        sys.exit(-1)

    dom1_train,word_map = get_data_matrix(args[0])
    dom2_train,word_map = get_data_matrix(args[1], word_map)

    dom1_len = dom1_train.shape[0]
    dom2_len = dom2_train.shape[0]

    # This dataset has 1000 instances per class, so 4000 for 2 datasets, and our method
    # reads positive instances first
    all_y = np.ones(dom1_len + dom2_len)
    all_y[dom1_len/2:dom1_len] = 2
    all_y[dom1_len+dom2_len/2:] = 2

    num_missing_feats = dom2_train.shape[1] - dom1_train.shape[1]
    dom1_train = hstack( (dom1_train, np.zeros( (dom1_len, num_missing_feats) ) ) )
    # dom2_train = dom2_train.tolil()[:, :dom1_train.shape[1]]
    all_train = vstack( (dom1_train, dom2_train) )
    ## Need this to match output format of cleartk (feature at index 1 is bias feature)
    all_train = hstack( (np.ones( (all_train.shape[0], 1) ), all_train))

    ## Add domain feature:
    d1f_ind = dom1_train.shape[1]
    d2f_ind = d1f_ind + 1
    domain_feats = lil_matrix( (all_train.shape[0], 2) )
    domain_feats[:dom1_len, 0] = 1
    domain_feats[dom1_len:, 1] = 1
    all_train = hstack( (all_train, domain_feats) )

    svml_file = open(join(args[2], 'training-data.liblinear'), 'w')
    # Since our previous files were not zero-based, we mimic that
    # so our downstream scripts work the same
    dump_svmlight_file(all_train, all_y, svml_file, zero_based=False)
    svml_file.close()

    # From here on out when writing indices we add 2 - one for the bias feature
    # that cleartk includes and one for the one-indexing cleartk writes.
    dom1_df = get_domain_feature(args[0])
    dom2_df = get_domain_feature(args[1])
    groups = {'Unigram':[], 'Bigram':[]}

    # Write feature lookup file that documents map of feature names to their indices
    with open(join(args[2], 'features-lookup.txt'), 'w') as lookup_f:
        lookup_f.write('%s : %d\n' % (dom1_df, d1f_ind+2))
        lookup_f.write('%s : %d\n' % (dom2_df, d2f_ind+2))

        ## Get domain types:
        for key,val in word_map.items():
            if len(key.split('_')) > 1:
                f_type = 'Bigram'
            else:
                f_type = 'Unigram'
            groups[f_type].append(val+2)    
            lookup_f.write('%s_%s : %d\n' % (f_type, key.replace(' ', '_'), val+2))    

    with open(join(args[2], 'feature-groups.txt'), 'w') as groups_f:
        groups_f.write('Domain : %d,%d\n' % (d1f_ind+2, d2f_ind+2))
        for f_type in ['Unigram', 'Bigram']:
            groups_f.write('%s : %s\n' % (f_type, ','.join(map(str, groups[f_type]))))


if __name__ == '__main__':
    main(sys.argv[1:])
