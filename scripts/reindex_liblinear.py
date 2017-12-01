#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
import sys
from os.path import dirname, join

from uda_common import read_feature_lookup, read_feature_groups

def main(args):
    if len(args) < 2:
        sys.stderr.write("Error: At least two required arguments: <Training input> <reduced training output file>\n")
        sys.exit(-1)

    X_train, y_train = load_svmlight_file(args[0])
    (num_instances, num_feats) = X_train.shape
    data_dir = dirname(args[0])
    out_file = args[1]
    out_dir = dirname(out_file)
    num_feats_allowed = int(out_dir.split("_")[-1].replace('k', '000'))
    sys.stderr.write("Argument indicates feature threshold of %d features allowed\n" % (num_feats_allowed))

    ## Find the low frequency feature we can discard:
    feat_counts = X_train.sum(0)

    for count in range(1,10):
        lowfreq_inds = np.where(feat_counts < count)[1]
        num_good_feats = num_feats - len(lowfreq_inds)
        sys.stderr.write("Found %d features that occurred >= %d times in the data\n" % (num_good_feats, count))
        if num_good_feats < num_feats_allowed:
            sys.stderr.write("Breaking at threshold %d\n" % (count))
            break

    ## Get the feature mapping, giving it an offset of -1 to adjust for the fact that
    ## java wrote liblinear with index 1 and load_svmlight_file gives it to us with
    ## index 1
    old_feat_map = read_feature_lookup(join(data_dir, 'features-lookup.txt'), offset=-1)
    old_groups = read_feature_groups(join(data_dir, 'feature-groups.txt'), offset=-1)
    new_groups = {}
    for domain in old_groups.keys():
        new_groups[domain] = []

    new_map_file = open(join(out_dir, 'reduced-features-lookup.txt'), 'w')

    ## Create a reduced feature matrix with only common features:
    new_X = np.matrix(np.zeros((num_instances, num_good_feats)))

    sys.stderr.write('Building matrix in reduced dimension space\n')
    new_ind = 0
    for ind in range(num_feats):
        feat_name = old_feat_map[ind]
        if not ind in lowfreq_inds:
            ## Use this feature:
            # 1) Write its mapping to feature lookup file: (unless it's the bias feature)
            if not ind == 0:
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
    new_group_file = open(join(out_dir, 'reduced-feature-groups.txt'), 'w')
    for feat_type in new_groups.keys():
        new_group_file.write('%s : %s\n' % (feat_type, ','.join(map(str,new_groups[feat_type]))))

    new_group_file.close()

    sys.stderr.write('Writing new svmlight file\n')
    f = open(out_file, 'w')
    dump_svmlight_file(new_X, y_train, f)
    f.close()

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
