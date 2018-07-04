#!/usr/bin/python
import sys
from os.path import join,exists,dirname

import numpy as np
from torch import nn
import torch.optim as optim
import torch
from torch.autograd import Function, Variable
from torch import FloatTensor
from sklearn.datasets import load_svmlight_file

from uda_common import read_feature_groups, read_feature_lookup, read_pivots

class MultiLayerPredictorModel(nn.Module):
    def __init__(self, input_features, pivot_features, hidden_nodes=200):
        super(MultiLayerPredictorModel, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_features, hidden_nodes))
        self.model.add_module('relu1', nn.ReLU(True))
        self.model.add_module('output_layer', nn.Linear(hidden_nodes, pivot_features))
        self.model.add_module('sigout', nn.Sigmoid())
    
    def forward(self, input):
        return self.model(input)

class DirectPredictorModel(nn.Module):
    def __init__(self, input_features, pivot_features):
        super(DirectPredictorModel, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_features, pivot_features))
        self.model.add_module('sigout', nn.Sigmoid())

    def forward(self, input):
        return self.model(input)
  
class NLLPredictorModel(nn.Module):
    def __init__(self, input_features, pivot_features, hidden_nodes=200):
        super(NLLPredictorModel, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_features, hidden_nodes))
        self.model.add_module('relu1', nn.ReLU(True))

        self.pivot_predictors = [] 
        for i in range(pivot_features):
            predictor = nn.Sequential()
            predictor.add_module('output_layer_%d' % i, nn.Linear(hidden_nodes, 2))
            self.pivot_predictors.append(predictor)

    def forward(self, input):
        feature = self.model(input)
        outputs = []
        for i in range( len(self.pivot_predictors) ):
            outputs.append(self.pivot_predictors[i](feature))

        return outputs

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required arguments: <data file> <pivot inds file> [backward=True|False]\n")
        sys.exit(-1)
   
    cuda = False 
    if torch.cuda.is_available():
        cuda = True

    if len(args) > 1:
        backward = bool(args[1])
        print("Direction is backward based on args=%s" % (args[1]))
    else:
        backward = False
        print("Direction is forward by default")

    # Define constants and hyper-params    
    goal_ind = 2
    training_portion = 0.8
    lr = 0.01
    epochs = 100
    batch_size = 50
    bce_loss = True
    nll_loss = False

    # can't be 2 loss functions
    assert not bce_loss or not nll_loss

    # Read the data:
    sys.stderr.write("Reading source data from %s\n" % (args[0]))
    # Ignore y from files since we're using pivot features as y
    all_X, _ = load_svmlight_file(args[0])
    # Read pivot values from X:
    pivot_map = read_pivots(args[1])
    pivot_inds = pivot_map.keys()
    pivot_inds.sort()
    # DEBUGGING
    # pivot_inds = [2,]
    num_pivots = len(pivot_inds)
    all_Y = all_X[:, pivot_inds].toarray()
    assert all_Y.max() == 1 and all_Y.min() == 0
    prevalences = all_Y.sum(0)
    print("Pivot prevalence: %s" % (str(prevalences)))
    # normalize y to -1/1 for softmarginloss
    if nll_loss:
        pass
    elif not bce_loss:
        all_Y = all_Y * 2 - 1

    # then zero out pivot features so they're not used to predict:
    all_X[:,pivot_inds] = 0

    num_instances, num_feats = all_X.shape

    domain_map = read_feature_groups(join(dirname(args[0]), 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(dirname(args[0]), 'reduced-features-lookup.txt'))


    direction = 1 if backward else 0
    sys.stderr.write("using domain %s as source, %s as target\n"  %
        (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

    # For this task, we don't care about source vs. target
    all_X[:, domain_inds[direction]] = 0
    all_X[:, domain_inds[1-direction]] = 0

    # select 80% of the data to train:
    train_inds = np.where( (np.random.rand(num_instances) < training_portion) == True)[0]
    dev_inds = np.setdiff1d( np.arange(num_instances), train_inds)
    num_train_instances = len(train_inds)
    train_X = all_X[train_inds,:]
    train_Y = all_Y[train_inds,:]
    dev_X = Variable(FloatTensor(all_X[dev_inds,:].toarray()))
    dev_Y = Variable(FloatTensor(all_Y[dev_inds,:]))

    model = DirectPredictorModel(num_feats, num_pivots)
#    model = MultiLayerPredictorModel(num_feats, num_pivots, hidden_nodes=500)

    if bce_loss:
        losses = [nn.BCELoss() for i in range(num_pivots)]
    elif nll_loss:
        majority = all_Y.shape[0] - prevalences
        majority_class_weights = prevalences / majority
        losses = [nn.NLLLoss(torch.FloatTensor([majority_class_weights[i], 1.])) for i in range(num_pivots)]
    else:
        losses = [nn.SoftMarginLoss() for i in range(num_pivots)]

    if cuda:
        dev_X = dev_X.cuda() 
        dev_Y = dev_Y.cuda()
        model.cuda()
        [loss_fn.cuda() for loss_fn in losses]

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in range( 1+ ( num_instances // batch_size ) ):
            start_ind = batch * batch_size
            if start_ind >= num_train_instances:
                #This happens if our number of instances is perfectly divisible by batch size (when batch_size=1 this is often).
                break
            end_ind = num_train_instances if start_ind + batch_size >= num_train_instances else start_ind+batch_size
            batch_X = Variable(FloatTensor(train_X[start_ind:end_ind,:].toarray()))
            batch_Y = Variable(FloatTensor(train_Y[start_ind:end_ind,:]))
            if cuda:
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()

            # model outputs are 0/1, fix to -1/1 for sigmoid loss
            preds = model.forward(batch_X)
            if not bce_loss:
                preds = preds * 2 - 1
            # 
            sum_loss = 0
            for out_ind in range(num_pivots):
                pivot_loss = losses[out_ind](preds[:,out_ind], batch_Y[:,out_ind])
                sum_loss = sum_loss + pivot_loss

            sum_loss.backward() 
            optimizer.step()
            epoch_loss += sum_loss.data
       
        # model outputs are 0/1, move to -1/1
        pred_valid = model.forward(dev_X)
        if not bce_loss:
            pred_valid = pred_valid * 2 - 1

        if bce_loss:
            tps = pred_valid * dev_Y
            true_preds = pred_valid
            true_gold = dev_Y
        else:
            tps = ((torch.sign(pred_valid)+1)/2)   *  ((torch.sign(dev_Y)+1)/2)
            true_preds = (torch.sign(pred_valid) + 1) / 2
            true_gold = (torch.sign(dev_Y) + 1 / 2).sum()

        
        precs = (tps.sum(0) / true_preds.sum(0)).data.cpu().numpy()
        prec_nans = np.where(np.isnan(precs))[0]
        if len(prec_nans) > 0:
            precs[prec_nans] = 0
        recalls = (tps.sum(0) / true_gold.sum(0)).data.cpu().numpy()
        rec_nans = np.where(np.isnan(recalls))[0]
        if len(rec_nans) > 0:
            recalls[rec_nans] = 0

        f1s = 2 * precs * recalls / (precs + recalls)
        f1_nans = np.where(np.isnan(f1s))[0]
        if len(f1_nans) > 0:
            f1s[f1_nans] = 0

        # accuracy gives credit for getting 0 and 1, where getting 0 is much easier than 1
        # since 1 is so rare. not as useful as f1 but closer to what the network is optimizing for
        if bce_loss:
            # if the difference is 0 they are the same -- so make all differences -1, add 1, to get
            # correct instances as 1s
            correct = -torch.abs(pred_valid - dev_Y) + 1
        else:
            # if the signs multiply to be positive they are correct, then convert from -1/1 to 0/1
            correct = ((torch.sign(pred_valid) * torch.sign(dev_Y) + 1) / 2)

        assert correct.max().data.cpu().numpy()[0] <= 1
        accuracies = correct.sum(0) / pred_valid.shape[0]
        total_acc = correct.sum() / (num_pivots * pred_valid.shape[0])
        print("Epoch %d has loss %f, p/r/f=%s/%s/%s total_acc=%f, with individual accuracies %s" % (epoch, epoch_loss, precs, recalls, f1s, total_acc, str(accuracies.data.cpu().numpy())))


if __name__ == '__main__':
    main(sys.argv[1:])

