#!/usr/bin/env python

import sys

import numpy as np
import torch
from learn_pivots_tm import PivotLearnerModel, StraightThroughLayer

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required arguments: <model file> [num pivots (100)]\n")
        sys.exit(-1)

    num_pivots = 100
    if len(args) > 1:
        num_pivots = int(args[1])

    model = torch.load(args[0])
    vec = np.abs(model.feature.input_layer.vector.data.cpu().numpy())
    inds = np.argsort(vec)
    pivot_inds = inds[0, -num_pivots:]
    pivot_inds.sort()
    for x in pivot_inds:
        print(x)

if __name__ == '__main__':
    main(sys.argv[1:])

