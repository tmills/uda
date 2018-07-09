#!/usr/bin/env python3
import sys
from os.path import join,exists,dirname
import random
from datetime import datetime
import time

import numpy as np
from numpy.random import randint
from sklearn.datasets import load_svmlight_file
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.optim as optim
import torch
from torch import FloatTensor

from uda_common import read_feature_groups, read_feature_lookup

# the concepts here come from: https://github.com/fungtion/DANN/blob/master/models/model.py
class GradientBlockerF(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # zero out the gradient so it doesn't affect any weights prior to this layer
        output = 0 * grad_output
        return output, None

# Instead of this, may be able to just regularize by forcing off-diagonal to zero
# didn't work bc/ of memory issues
class StraightThroughLayer(nn.Module):
    def __init__(self, input_features):
        super(StraightThroughLayer, self).__init__()
        self.vector = nn.Parameter( torch.ones(1, input_features) )

    def forward(self,  input_data):
        output = torch.mul(input_data, self.vector)
        return output
        

class PivotLearnerModel(nn.Module):

    def __init__(self, input_features):
        super(PivotLearnerModel, self).__init__()
        # Feature takes you from input to the "representation"
        self.feature = nn.Sequential()
        # straight through layer just does an element-wise product with a weight vector
        num_features = input_features
        # self.vector = nn.Parameter( torch.randn(1, input_features) )
        self.feature.add_module('input_layer', StraightThroughLayer(input_features))
        # Standard feed forward layer:
        # num_features = 200
        # self.feature.add_module('input_layer', nn.Linear(input_features, num_features))
        # self.feature.add_module('relu', nn.ReLU(True))

        # task_classifier maps from a feature representation to a task prediction
        self.task_classifier = nn.Sequential()
        self.task_classifier.add_module('task_binary', nn.Linear(num_features, 1))
        self.task_classifier.add_module('task_sigmoid', nn.Sigmoid())
        
        # domain classifier maps from a feature representation to a domain prediction
        self.domain_classifier = nn.Sequential()
        # hidden_nodes = 100
        # self.domain_classifier.add_module('domain_hidden', nn.Linear(num_features, hidden_nodes, bias=False))
        # self.domain_classifier.add_module('relu', nn.ReLU(True))

        # No bias because bias could learn data prevalence
        self.domain_classifier.add_module('domain_classifier', nn.Linear(num_features, 1, bias=False))
        # # self.domain_classifier.add_module('domain_predict', nn.Linear(100, 1))
        self.domain_classifier.add_module('domain_sigmoid', nn.Sigmoid())
        
        # self.domain_classifier2 = nn.Sequential()
        # self.domain_classifier2.add_module('domain_linear', nn.Linear(num_features, 1, bias=False))
        # # # self.domain_classifier.add_module('domain_predict', nn.Linear(100, 1))
        # self.domain_classifier2.add_module('domain_sigmoid', nn.Sigmoid())

    def forward(self, input_data):
        feature = self.feature(input_data)

        # Get task prediction
        task_prediction = self.task_classifier(feature)

        # Get domain prediction
        domain_prediction = self.domain_classifier( GradientBlockerF.apply(feature) )

        # Copy weights over to the confusion side (and don't modify them)
        # self.confusion_layer.weight.data = self.domain_classifier.domain_classifier.weight.data
        domain_weights = Variable(self.domain_classifier.domain_classifier.weight.data, requires_grad=False)
        # domain_bias = Variable(self.domain_classifier.domain_classifier.bias.data, requires_grad=False)

        # confused_prediction = nn.functional.sigmoid( torch.dot(domain_weights, feature) ) 
        confused_prediction = nn.functional.sigmoid( torch.matmul(domain_weights, feature.t()) )

        return task_prediction, domain_prediction, confused_prediction

def confusion_loss_fn(system, gold):
    # doesn't work
    # return torch.abs(system - 0.5)
    # doesn't work -- just makes things close to 0.5
    # return torch.pow(system - 0.5, 2)
    return nn.functional.binary_cross_entropy(1-system, gold)

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required arguments: <data file> [backward True|False]\n")
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

    # constants:
    goal_ind = 2
    domain_weight = 0.1
    confusion_weight = 0.1
    reg_weight = 0.1
    lr = 0.01
    epochs = 10000

    date_str = datetime.now().isoformat()

    # Read the data:
    sys.stderr.write("Reading source data from %s\n" % (args[0]))
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
    num_train_instances = int(X_source.shape[0] * 0.8)
    X_task_train = X_source[:num_train_instances,:]
    y_task_train = y_source[:num_train_instances]
    X_task_valid = X_source[num_train_instances:, :]
    y_task_valid = y_source[num_train_instances:]

    target_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
    X_target = all_X[target_instance_inds,:]
    X_target[:, domain_inds[direction]] = 0
    X_target[:, domain_inds[1-direction]] = 0
    num_target_train = int(X_target.shape[0] * 0.8)
    X_target_train = X_target[:num_target_train,:]
    X_target_valid = X_target[num_target_train:, :]
    num_target_instances = X_target_train.shape[0]

    model = PivotLearnerModel(num_feats)
    task_loss_fn = nn.BCELoss()
    domain_loss_fn = nn.BCELoss()
    l1_loss = nn.L1Loss()
    

    if cuda:
        model.cuda()
        task_loss_fn.cuda()
        domain_loss_fn.cuda()

    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    # Main training loop
    inds = np.arange(num_train_instances)
    best_qualifying_f1 = 0

    for epoch in range(epochs):
        epoch_loss = 0
        random.shuffle(inds)

        epoch_start = time.time()
        # Do a training epoch:
        for source_ind in inds:
            model.zero_grad()

            # standardized_X = (X_train[source_ind,:].toarray() - X_mean) / X_std
            source_batch = Variable(FloatTensor(X_task_train[source_ind,:].toarray()))# read input
            source_task_labels = Variable(torch.unsqueeze(FloatTensor([y_task_train[source_ind],]), 1))# read task labels
            source_domain_labels = Variable(torch.unsqueeze(FloatTensor([0.,]), 1)) # set to 0

            if cuda:
                source_batch = source_batch.cuda()
                source_task_labels = source_task_labels.cuda()
                source_domain_labels = source_domain_labels.cuda()
            
            # Get the task loss and domain loss for the source instance:
            task_out, source_domain_out, source_confusion_out = model.forward(source_batch)
            task_loss = task_loss_fn(task_out, source_task_labels)
            domain_loss = domain_loss_fn(source_domain_out, source_domain_labels)
            confusion_loss = confusion_loss_fn(source_confusion_out, source_domain_labels)
            reg_term = l1_loss(model.feature.input_layer.vector, torch.zeros_like(model.feature.input_layer.vector))

            # Randomly select a target instance:
            target_ind = randint(num_target_instances)
            target_batch = Variable(FloatTensor(X_target_train[target_ind,:].toarray())) # read input
            target_domain_labels = Variable(torch.unsqueeze(FloatTensor([1.,]), 1)) # set to 1

            if cuda:
                target_batch = target_batch.cuda()
                target_domain_labels = target_domain_labels.cuda()
            
            # Get the domain loss for the target instances:
            _, target_domain_out, target_confusion_out = model.forward(target_batch)
            target_domain_loss = domain_loss_fn(target_domain_out, target_domain_labels)
            target_confusion_loss = confusion_loss_fn(target_confusion_out, target_domain_labels)

            # Get sum loss update weights:
            # domain adaptation:
            total_loss = (task_loss + 
                            domain_weight * (domain_loss + target_domain_loss) + 
                            confusion_weight * (confusion_loss + target_confusion_loss) +
                            reg_weight * reg_term)

            total_loss.backward()
            epoch_loss += total_loss

            optimizer.step()

        # At the end of every epoch, examine domain accuracy and how many non-zero parameters we have
        source_eval_X = X_task_valid
        source_eval_y = y_task_valid
        source_task_out, source_domain_out, source_confusion_out = model.forward( Variable(FloatTensor(source_eval_X.toarray())).cuda() )
        # If this goes down that means its getting more confused about the domain
        domain_out_stdev = source_domain_out.std()

        # source domain is 0, count up predictions where 1 - prediction = 1
        source_domain_preds = np.round(source_domain_out.cpu().data.numpy())
        source_predicted_count = np.sum(1 - source_domain_preds)


        target_eval_X = X_target_valid
        _, target_domain_out, target_confusion_out = model.forward( Variable(FloatTensor(target_eval_X.toarray())).cuda() )
        # if using sigmoid output (0/1) with BCELoss
        target_domain_preds = np.round(target_domain_out.cpu().data.numpy())
        target_predicted_count = np.sum(target_domain_preds)

        domain_acc = (source_predicted_count + target_predicted_count) / (source_eval_X.shape[0] + target_eval_X.shape[0])

        # predictions of 1 are the positive class: tps are where prediction and gold are 1
        source_y_pred = np.round(source_task_out.cpu().data.numpy()[:,0])
        tps = np.sum(source_y_pred * source_eval_y)
        true_preds = source_y_pred.sum()
        true_labels = source_eval_y.sum()

        recall = tps / true_labels
        prec = 1 if tps == 0 else tps / true_preds
        f1 = 2 * recall * prec / (recall+prec)

        try:
            weights = model.feature.input_layer.vector
            num_zeros = (weights.data==0).sum() 
            near_zeros = (torch.abs(weights.data)<0.000001).sum()

            print("Min (abs) weight: %f" % (torch.abs(weights).min()))
            print("Max (abs) weight: %f" % (torch.abs(weights).max()))
            print("Ave weight: %f" % (torch.abs(weights).mean()))
        except:
            num_zeros = near_zeros = -1

        epoch_len = time.time() - epoch_start
        print("[Source] Epoch %d [%0.1fs]: loss=%f\tnear_zero=%d\tnum_insts=%d\tdom_acc=%f\tdom_std=%f\tP=%f\tR=%f\tF=%f" % (epoch, epoch_len, epoch_loss, near_zeros, len(source_eval_y), domain_acc, domain_out_stdev, prec, recall, f1))

        if f1 > 0.8 and f1 > best_qualifying_f1 and abs(domain_acc - 0.5) < 0.05:
            print("This model is the most accurate-to-date that is confused between domains so we're writing it.")
            best_qualifying_f1 = f1
            torch.save(model, 'model_epoch%04d_dt=%s.pt' % (epoch, date_str))

if __name__ == '__main__':
    main(sys.argv[1:])

