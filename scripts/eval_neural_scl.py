#!/usr/bin/python
import sys
from os.path import join,exists,dirname
from predict_pivots import MultiLayerPredictorModel

import numpy as np
from torch import nn
import torch.optim as optim
import torch
from torch.autograd import Function, Variable
from torch import FloatTensor
from sklearn.datasets import load_svmlight_file

from uda_common import read_feature_groups, read_feature_lookup, read_pivots

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_features, hidden_nodes):
        super(MultiLayerPerceptron, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_features, hidden_nodes))
        self.model.add_module('relu', nn.ReLU(True))
        self.model.add_module('hidden_layer', nn.Linear(hidden_nodes, 1))
        self.model.add_module('sigmoid', nn.Sigmoid())
    
    def forward(self, input):
        return self.model(input)

class PivotRepresentationExtractor(nn.Module):
    def __init__(self, pivot_predictor):
        super(PivotRepresentationExtractor, self).__init__()
        input_features = pivot_predictor.model.input_layer.in_features
        hidden_nodes = pivot_predictor.model.input_layer.out_features
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_features, hidden_nodes))
        self.model.add_module('relu1', nn.ReLU(True))
        self.model.input_layer.weight.data = pivot_predictor.model.input_layer.weight.data

    def forward(self, input):
        return self.model(input)

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required arguments: <data file> <pivot predictor model file> [backward True|False]\n")
        sys.exit(-1)
   
    cuda = False
    training_portion = 0.8 
    goal_ind = 2
    reg_weight = 0.1
    epochs = 10000
    hidden_nodes = 200
    batch_size = 36

    if torch.cuda.is_available():
        cuda = True
    
    if len(args) > 2:
        backward = bool(args[2])
        print("Direction is backward based on args=%s" % (args[2]))
    else:
        backward = False
        print("Direction is forward by default")

    pivot_predictor = torch.load(args[1])
    pivot_predictor.cpu()
    pivot_rep_model = PivotRepresentationExtractor(pivot_predictor)

    all_X, all_y = load_svmlight_file(args[0])
    
    # y is 1,2 by default, map to -1,1 for sigmoid training
    all_y -= 1   # 0/1

    num_instances, num_feats = all_X.shape

    domain_map = read_feature_groups(join(dirname(args[0]), 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(dirname(args[0]), 'reduced-features-lookup.txt'))

    direction = 1 if backward else 0
    sys.stderr.write("using domain %s as source, %s as target\n"  %
        (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

    source_instance_inds = np.where(all_X[:,domain_inds[direction]].toarray() > 0)[0]
    X_source = all_X[source_instance_inds,:]
    X_source[:, domain_inds[direction]] = 0
    X_source[:, domain_inds[1-direction]] = 0
    y_source = all_y[source_instance_inds]
    num_source_instances = X_source.shape[0]

    train_inds = np.where( (np.random.rand(num_source_instances) < training_portion) == True)[0]
    dev_inds = np.setdiff1d( np.arange(num_source_instances), train_inds)
    num_train_instances = len(train_inds)
    train_X = X_source[train_inds,:]
    train_y = y_source[train_inds]
    dev_X = Variable(FloatTensor(X_source[dev_inds,:].toarray()))
    dev_y = Variable(FloatTensor(y_source[dev_inds]))

    pivot_prediction_features = pivot_rep_model.model.input_layer.out_features
    model = MultiLayerPerceptron(train_X.shape[1] + pivot_prediction_features, hidden_nodes)
    loss_fn = nn.BCELoss()
    l1_loss = nn.L1Loss()

    if cuda:
        dev_X = dev_X.cuda() 
        dev_y = dev_y.cuda()
        model.cuda()
        pivot_rep_model.cuda()
        loss_fn.cuda()
        l1_loss.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)

    best_f = 0
    # Training loop:
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch in range( 1+ ( num_instances // batch_size ) ):
            model.zero_grad()
            start_ind = batch * batch_size
            if start_ind >= num_train_instances:
                #This happens if our number of instances is perfectly divisible by batch size (when batch_size=1 this is often).
                break
            end_ind = num_train_instances if start_ind + batch_size >= num_train_instances else start_ind+batch_size
            batch_X = Variable(FloatTensor(train_X[start_ind:end_ind,:].toarray()))
            batch_y = Variable(FloatTensor(train_y[start_ind:end_ind]))

            if cuda:
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            pivot_reps = pivot_rep_model.forward(batch_X)
            # we don't need to cmpute gardients wrt this variable, we're just using it as an input to another network:
            pivot_reps = pivot_reps.detach()

            full_X = torch.cat( (batch_X, pivot_reps), 1)
            predict_y = model.forward(full_X)

            train_loss = loss_fn(predict_y, batch_y)

            reg_loss = 0
            for param in model.parameters():
                reg_param = l1_loss(param, torch.zeros_like(param))
                reg_loss = reg_loss + reg_param

            loss = train_loss + reg_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data
        
        pivot_reps = pivot_rep_model.forward(dev_X)
        pivot_reps = pivot_reps.detach()
        full_dev_X = torch.cat( (dev_X, pivot_reps), 1)
        predict_y = model.forward(full_dev_X).round()
        tps = dev_y * predict_y[:,0]
        tps_sum = tps.sum().cpu().data.numpy()[0]
        predict_pos = predict_y.sum().cpu().data.numpy()[0]
        gold_pos = dev_y.sum().cpu().data.numpy()[0]
        prec = tps_sum  / predict_pos
        rec = tps_sum / gold_pos
        if prec == 0.0:
            prec = 1.0
        f1 = 2 * prec * rec / (prec + rec)
        print("Epoch %d loss is %f, p/r/f=%f/%f/%f" % (epoch, epoch_loss, prec, rec, f1))

        if f1 > best_f:
            torch.save(model, 'best_neural_scl_model.pt')
            best_f = f1

        
    # Load best model:
    model = torch.load('best_neural_scl_model.pt')

    # Eval:
    target_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
    X_target = all_X[target_instance_inds,:]
    X_target[:, domain_inds[direction]] = 0
    X_target[:, domain_inds[1-direction]] = 0
    num_target_train = int(X_target.shape[0] * 0.8)
    X_target_train = X_target[:num_target_train,:]
    X_target_valid = X_target[num_target_train:, :]
    num_target_instances = X_target_train.shape[0]

if __name__ == '__main__':
    main(sys.argv[1:])
