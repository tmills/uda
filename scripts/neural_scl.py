#!/usr/bin/python
import sys
from os.path import join,exists,dirname

import numpy as np
from numpy.random import randint
from sklearn.datasets import load_svmlight_file
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.optim as optim
import torch
from torch import FloatTensor

from uda_common import read_feature_groups, read_feature_lookup, read_pivot_file

class PredictorWithPivots(nn.Module):
    def __init__(self, input_features, pivot_model, pivots, hidden_nodes=200):
        self.num_features = input_features
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_features, hidden_nodes))
        self.model.add_module('input_relu', nn.Relu(True))

        self.mask = torch.ones(self.num_features)
        self.mask[pivots] = 0
        self.pivot_model = pivot_model
        self.pivots = pivots
        num_pivots = len(pivots)

        self.classifier = nn.Sequential()
        self.classifier.add_module('output_layer', nn.Linear(hidden_nodes + num_pivots , 1))
        self.classifier.add_module('output_sigmoid', nn.Sigmoid())

    def forward(self, input_data):
        normal_hidden = self.model(input_data)
        masked_input = input_data * self.mask
        pivot_preds = self.pivot_model(masked_input)
        cat_input = torch.cat( (normal_hidden, pivot_preds) )
        output = self.classifier(cat_input)
        return output


def main(args):
    if len(args) < 3:
        sys.stderr.write("Required arguments: <data file> <pivot data file> <pivot model file> [backward True|False]\n")
        sys.exit(-1)
   
    lr = 0.01
    epochs = 100

    cuda = False 
    if torch.cuda.is_available():
        cuda = True
    
    if len(args) > 3:
        backward = bool(args[3])
        print("Direction is backward based on args=%s" % (args[3]))
    else:
        backward = False
        print("Direction is forward by default")

    # Read the data:
    goal_ind = 2

    sys.stderr.write("Reading source data from %s\n" % (args[0]))
    all_X, all_y = load_svmlight_file(args[0])

    # TODO: Necessary?
    # y is 1,2 by default, map to -1,1 for sigmoid training
    all_y -= 1   # 0/1

    num_instances, num_feats = all_X.shape
    domain_map = read_feature_groups(join(dirname(args[0]), 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']
    feature_map = read_feature_lookup(join(dirname(args[0]), 'reduced-features-lookup.txt'))

    direction = 1 if backward else 0
    sys.stderr.write("using domain %s as source, %s as target\n"  %
        (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

    pivot_inds = read_pivot_file(args[1])
    pivot_predictor_model = torch.load(args[2])

    model = PredictorWithPivots(num_feats, pivot_predictor_model, pivot_inds)
    loss = nn.SoftMarginLoss()

    if cuda:
        model.cuda()
        loss_fn.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    
    ## Copy/paste epoch loop from updated predict pivots.
    #         

if __name__ == '__main__':
    main(sys.argv[1:])

