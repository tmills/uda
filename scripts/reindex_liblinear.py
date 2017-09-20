#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
import sys
from os.path import dirname, join

from uda_common import read_feature_lookup, read_feature_groups

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error: At least one required argument: <Filename>*\n")
        sys.exit(-1)

    X_train, y_train = load_svmlight_file(args[0])
    (num_instances, num_feats) = X_train.shape
    data_dir = dirname(args[0])

    ## Find the low frequency feature we can discard:
    feat_counts = X_train.sum(0)
    lowfreq_inds = np.where(feat_counts < 3)[1]
    sys.stderr.write("Found %d features that occurred < 3 times in the data\n" % len(lowfreq_inds))

    ## Get the feature mapping, giving it an offset of -1 to adjust for the fact that
    ## java wrote liblinear with index 1 and load_svmlight_file gives it to us with
    ## index 1
    old_feat_map = read_feature_lookup(join(data_dir, 'features-lookup.txt'), offset=-1)
    old_groups = read_feature_groups(join(data_dir, 'feature-groups.txt'), offset=-1)
    new_groups = {}
    for domain in old_groups.keys():
        new_groups[domain] = []

    new_map_file = open(join(data_dir, 'reduced-features-lookup.txt'), 'w')

    ## Create a reduced feature matrix with only common features:
    num_good_feats = num_feats - len(lowfreq_inds)
    new_X = np.matrix(np.zeros((num_instances, num_good_feats)))

    sys.stderr.write('Building matrix in reduced dimension space\n')
    new_ind = 0
    for ind in range(num_feats):
        feat_name = old_feat_map[ind]
        if not ind in lowfreq_inds:
            ## Use this feature:
            # 1) Write its mapping to feature lookup file:
            new_map_file.write('%s : %d\n' % (feat_name, new_ind))

            # 2) Add its column to the data matrix:
            new_X[:,new_ind] += X_train[:,ind]

            # 3) Add its index to the group mapping file:
            for feat_type in old_groups.keys():
                if ind in old_groups[feat_type]:
                    new_groups[feat_type].append(new_ind)

            new_ind += 1

    new_map_file.close()

    sys.stderr.write('Writing reduced feature groups file\n')
    new_group_file = open(join(data_dir, 'reduced-feature-groups.txt'), 'w')
    for feat_type in new_groups.keys():
        new_group_file.write('%s : %s\n' % (feat_type, ','.join(map(str,new_groups[feat_type]))))

    new_group_file.close()

    sys.stderr.write('Writing new svmlight file\n')
    dump_svmlight_file(new_X, y_train, sys.stdout)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
