#!/usr/bin/env python
import numpy as np
import sys
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import scipy.sparse
import scipy.stats
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c

def main(args):
    if len(args) < 2:
        sys.stderr.write("Two required arguments: <source data> <target data>\n")
        sys.exit(-1)

    source_X, source_y = load_svmlight_file(args[0])
    num_source_instances, num_features = source_X.shape
    target_X, target_y = load_svmlight_file(args[1])
    target_X = align_test_X_train(source_X, target_X)
    num_target_instances, num_target_features = target_X.shape

    ## First adaptation variable: Domain identification performance
    combined_X = scipy.sparse.lil_matrix(np.zeros((num_source_instances+num_target_instances, num_features)))
    domain_y = np.zeros(num_source_instances+num_target_instances)
    domain_y[:num_source_instances] = 1
    c, acc = find_best_c(combined_X, domain_y, scorer=accuracy_score)
    print("Best performance for telling domains apart is %f" % (acc))

    ## Second adaptation variable: cosine similarity between vector averages
    source_ave = np.average(source_X.toarray(), 0)
    target_ave = np.average(target_X.toarray(), 0)
    dot_sim = source_ave.dot(target_ave)
    # This function computes kl-divergence if given 2 arguments and does the normalizatino for you:
    kl = scipy.stats.entropy(source_ave, target_ave)
    print("Raw dot product is %f, kl divergence is %f" % (dot_sim, kl))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
