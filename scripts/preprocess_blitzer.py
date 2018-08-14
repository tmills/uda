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

def get_domain_feature(dir):
    domain_feature = basename(dir)
    if domain_feature == '':
        domain_feature = basename(dirname(dir))
    
    return 'Domain_' + domain_feature

def parse_raw_domain(dir):
    data = {'data':[], 'target':[], 'target_names':['negative, positive']}
    
    texts = []
    labels = []
    for polarity in ['positive', 'negative']:
        # We need to read the text and add an outer tag because the
        # dataset is not properly formatted xml
        fn = join(dir, '%s.review' % polarity)
        f = open(fn, 'r')
        text = ('<doc>\n' + f.read() + '</doc>').replace('&', '&amp;').replace('\x1a', ' ')
        lines = text.split('\n')
        tree = ET.fromstring(text)
        # root = tree.getroot()
        for review in tree.findall('review'):
            # Extract the rating and give it a polarity label (0=negative, 1=positive)
            rating = float(review.find('rating').text)

            if rating < 3:
                label = 0
            elif rating > 3:
                label = 1
            else:
                # 3 is ambiguous so they skip  those
                continue
            labels.append(label)
            text = review.find('review_text')
            texts.append(text.text.strip())
            # texts.append(text.text.strip() + ' %s' % (domain_feature))
    
    return texts, labels      


def main(args):
    if len(args) < 3:
        sys.stderr.write('Error: 3 required arguments: <domain 1> <domain 2> <output dir>')

    dom1_df = get_domain_feature(args[0])
    dom2_df = get_domain_feature(args[1])
    dom1_text, dom1_labels = parse_raw_domain(args[0])
    dom2_text, dom2_labels = parse_raw_domain(args[1])
    all_y = np.concatenate((dom1_labels, dom2_labels))

    count_vect = CountVectorizer(ngram_range=(1,2))
    count_vect.fit(dom1_text + dom2_text)

    dom1_train_counts = count_vect.transform(dom1_text)
    dom2_train_counts = count_vect.transform(dom2_text)
    all_X = vstack( (dom1_train_counts, dom2_train_counts) )
    all_X = hstack( (np.ones( (all_X.shape[0], 1) ), all_X))

    dom1_len = dom1_train_counts.shape[0]

    d1_scores = cross_validate(SGDClassifier(loss='modified_huber'),
                    dom1_train_counts,
                    all_y[:dom1_len],
                    cv=5)
    d1_mean = d1_scores['test_score'].mean()

    d2_scores = cross_validate(SGDClassifier(loss='modified_huber'),
                    dom2_train_counts,
                    all_y[dom1_len:],
                    cv=5)
    d2_mean = d2_scores['test_score'].mean()

    print("Unmodified in-domain 5-fold CV performance: dom1=%f, dom2=%f" % (d1_mean, d2_mean))

    d1f_ind = dom1_train_counts.shape[1]
    d2f_ind = d1f_ind + 1
    domain_feats = lil_matrix( (all_X.shape[0], 2) )
    domain_feats[:dom1_len, 0] = 1
    domain_feats[dom1_len:, 1] = 1
    all_X = hstack( (all_X, domain_feats) )

    svml_file = open(join(args[2], 'training-data.liblinear'), 'w')
    # Since our previous files were not zero-based, we mimic that
    # so our downstream scripts work the same
    dump_svmlight_file(all_X, all_y, svml_file, zero_based=False)
    svml_file.close()

    # From here on out when writing indices we add 2 - one for the bias feature
    # that cleartk includes and one for the one-indexing cleartk writes.
    lookup_f = open(join(args[2], 'features-lookup.txt'), 'w')
    lookup_f.write('%s : %d\n' % (dom1_df, d1f_ind+2))
    lookup_f.write('%s : %d\n' % (dom2_df, d2f_ind+2))

    groups = {'Unigram':[], 'Bigram':[]}
    for key,val in count_vect.vocabulary_.items():
        if len(key.split()) > 1:
            f_type = 'Bigram'
        else:
            f_type = 'Unigram'
        
        groups[f_type].append(val+2)        
        lookup_f.write('%s_%s : %d\n' % (f_type, key.replace(' ', '_'), val+2))

    lookup_f.close()
    # now write feature groups file
    groups_f = open(join(args[2], 'feature-groups.txt'), 'w')
    groups_f.write('Domain : %d,%d\n' % (d1f_ind+2, d2f_ind+2))
    for f_type in ['Unigram', 'Bigram']:
        groups_f.write('%s : %s\n' % (f_type, ','.join(map(str, groups[f_type]))))

    groups_f.close()

if __name__ == '__main__':
    main(sys.argv[1:])
