#!/usr/bin/env python
import numpy as np
import sys
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import scipy.sparse
import scipy.stats
from uda_common import zero_pivot_columns, zero_nonpivot_columns, read_pivots, evaluate_and_print_scores, align_test_X_train, get_f1, find_best_c, get_preds
from sklearn import svm

def main(args):
    if len(args) < 3:
        sys.stderr.write("Two required arguments: <source data> <target data> <pivot index file>\n")
        sys.exit(-1)

    goal_ind = 2
    source_X, source_y = load_svmlight_file(args[0])
    num_source_instances, num_features = source_X.shape
    target_X, target_y = load_svmlight_file(args[1])
    target_X = align_test_X_train(source_X, target_X)
    num_target_instances, num_target_features = target_X.shape

    ## First adaptation variable: Domain identification performance
    combined_X = scipy.sparse.lil_matrix(np.zeros((num_source_instances+num_target_instances, num_features)))
    combined_X[:num_source_instances,:] += source_X
    combined_X[num_source_instances:,:] += target_X
    domain_y = np.zeros(num_source_instances+num_target_instances)
    domain_y[:num_source_instances] = 1
    c, acc = find_best_c(combined_X, domain_y, scorer=accuracy_score)
    print("Best performance for telling domains apart is %f" % (acc))

    ## Second adaptation variable: cosine similarity between vector averages
    source_ave = np.average(source_X.toarray(), 0)
    target_ave = np.average(target_X.toarray(), 0)
    dot_sim = source_ave.dot(target_ave)
    ### Normalize and re-do:
    source_ave /= source_ave.sum()
    target_ave /= target_ave.sum()
    cos_sim = source_ave.dot(target_ave)

    # This function computes kl-divergence if given 2 arguments and does the normalization for you:
    kl = scipy.stats.entropy(source_ave+0.01, target_ave+0.01)
    print("Raw dot product is %f, cos sim is %f, kl divergence is %f" % (dot_sim, cos_sim, kl))

    ## Get prevalences for source and target
    c, acc = find_best_c(source_X, source_y, scorer_args={'goal_ind':goal_ind})
    target_preds = get_preds(source_X, source_y, target_X, C=c)
    gold_prev = len(np.where(source_y==goal_ind)[0]) / float(len(source_y))
    target_prev = len(np.where(target_preds==goal_ind)[0]) / float(len(target_preds))
    print("Gold goal prevalence is %f compared to predicted target prevalence of %f" % (gold_prev, target_prev))

    # Last variable is how well can we predict pivot features with non-pivot features:
    # pivots = read_pivots(args[2])
    # nopivot_X_train = zero_pivot_columns(combined_X, pivots).toarray()
    # accuracies = np.zeros(len(pivots))
    #
    # for ind,pivot in enumerate(pivots):
    #     ## build a test set with pivot values:
    #     pivot_y = np.zeros(combined_X.shape[0]) + combined_X[:,pivot].toarray()[:,0]
    #     if not len(pivot_y) == (len(np.where(pivot_y == 1.0)[0]) + len(np.where(pivot_y == 0.0)[0])):
    #         print("Feature %d is not restricted to 0/1 so we are skipping" % (pivot))
    #         continue
    #     c, acc = find_best_c(nopivot_X_train, pivot_y, scorer=accuracy_score, C_list = [1.0])
    #     accuracies[ind] = acc
    #     print("Debugging: Accuracy for pivot %d is %f" % (ind, acc))
    #
    # ave_acc = accuracies.mean()
    # med_acc = np.median(accuracies)
    # print("Average accuracy at pivot prediction is %f, median accuracy is %f" %(ave_acc, med_acc))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
