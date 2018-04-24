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

from uda_common import read_feature_groups, read_feature_lookup

# the concepts here come from: https://github.com/fungtion/DANN/blob/master/models/model.py
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * 0 #grad_output.neg() * ctx.alpha
        return output, None

class TwoOutputModel(nn.Module):

    def __init__(self, input_features, hidden_nodes, num_outputs):
        super(TwoOutputModel, self).__init__()
        # Feature takes you from input to the "representation"
        self.feature = nn.Sequential()
        self.feature.add_module('input_layer', nn.Linear(input_features, hidden_nodes))
        self.feature.add_module('relu', nn.ReLU(True))
        # self.feature.add_module('hidden_layer', nn.Linear(hidden_nodes, hidden_nodes))
        # self.feature.add_module('relu2', nn.ReLU(True))

        # task_classifier maps from a feature representation to a task prediction
        self.task_classifier = nn.Sequential()
        # self.task_classifier.add_module('task_linear', nn.Linear(hidden_nodes, hidden_nodes))
        # self.task_classifier.add_module('relu2', nn.ReLU(True))
        if num_outputs > 2:
            self.task_classifier.add_module('task_linear', nn.Linear(hidden_nodes, num_outputs))
            self.task_classifier.add_module('task_softmax', nn.LogSoftmax())
        else:
            self.task_classifier.add_module('task_binary', nn.Linear(hidden_nodes, 1))
            self.task_classifier.add_module('task_sigmoid', nn.Sigmoid())
        
        # domain classifier maps from a feature representation to a domain prediction
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('domain_linear', nn.Linear(hidden_nodes, 1))
        # # self.domain_classifier.add_module('domain_predict', nn.Linear(100, 1))
        self.domain_classifier.add_module('domain_sigmoid', nn.Sigmoid())

    def forward(self, input_data, alpha):
        ## standardize input to -1/1
        # input_data = input_data * 2 - 1
        feature = self.feature(input_data)
        task_prediction = self.task_classifier(feature)

        # Get domain prediction
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_prediction = self.domain_classifier(reverse_feature)

        return task_prediction, domain_prediction

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required arguments: <data file> [backward True|False]\n")
        sys.exit(-1)
    
    if torch.cuda.is_available():
        cuda = True
    
    if len(args) > 1:
        backward = bool(args[1])
        print("Direction is backward based on args=%s" % (args[1]))
    else:
        backward = False
        print("Direction is forward by default")

    # Read the data:
    goal_ind = 2
    domain_weight = 0.1

    sys.stderr.write("Reading source data from %s\n" % (args[0]))
    all_X, all_y = load_svmlight_file(args[0])
    
    # y is 1,2 by default, map to -1,1 for sigmoid training
    all_y -= 1   # 0/1
    # all_y *= 2   # 0/2
    # all_y -= 1   # -1/1

    num_instances, num_feats = all_X.shape

    domain_map = read_feature_groups(join(dirname(args[0]), 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(dirname(args[0]), 'reduced-features-lookup.txt'))

    # Configure gloabel network params:
    lr = 0.01
    num_hidden_nodes = 1000
    epochs = 100

    direction = 1 if backward else 0
    sys.stderr.write("using domain %s as source, %s as target\n"  %
        (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

    train_instance_inds = np.where(all_X[:,domain_inds[direction]].toarray() > 0)[0]
    X_train = all_X[train_instance_inds,:]
    X_train[:, domain_inds[direction]] = 0
    X_train[:, domain_inds[1-direction]] = 0
    # X_mean = X_train.mean(0)
    # X_std = np.std(X_train.toarray(),0)
    # zeros = np.where(X_std == 0)[0]
    # X_std[zeros] = 1   ## If we divide by 1 later nothing changes.

    y_train = all_y[train_instance_inds]
    num_train_instances = X_train.shape[0]

    test_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
    X_test = all_X[test_instance_inds,:]
    X_test[:, domain_inds[direction]] = 0
    X_test[:, domain_inds[1-direction]] = 0
    y_test = all_y[test_instance_inds]
    num_test_instances = X_test.shape[0]
    
    model = TwoOutputModel(num_feats, num_hidden_nodes, 2)
    task_loss_fn = nn.BCELoss()
    domain_loss_fn = nn.BCELoss()

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if cuda:
        model.cuda()
        task_loss_fn.cuda()
        domain_loss_fn.cuda()

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        selected_source_inds = []
        # Do a training epoch:
        for ind in range(num_train_instances):
            model.zero_grad()

            ## Gradually increase (?) the importance of the regularization term
            p = float(ind + epoch * num_train_instances*2) / (epochs * num_train_instances*2)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            ## Randomly select a training instance:
            source_ind = randint(num_train_instances)
            selected_source_inds.append(source_ind)
            # standardized_X = (X_train[source_ind,:].toarray() - X_mean) / X_std
            source_batch = Variable(FloatTensor(X_train[source_ind,:].toarray()))# read input
            source_task_labels = Variable(FloatTensor([y_train[source_ind],]))# read task labels
            source_domain_labels = Variable(FloatTensor([0.,])) # set to 0

            if cuda:
                source_batch = source_batch.cuda()
                source_task_labels = source_task_labels.cuda()
                source_domain_labels = source_domain_labels.cuda()
            
            # Get the task loss and domain loss for the source instance:
            task_out, domain_out = model.forward(source_batch, alpha)
            task_loss = task_loss_fn(task_out, source_task_labels)
            domain_loss = domain_loss_fn(domain_out, source_domain_labels)

            # Randomly select a target instance:
            target_ind = randint(num_test_instances)
            # # standardized_X = (X_test[target_ind,:].toarray() - X_mean) / X_std
            target_batch = Variable(FloatTensor(X_test[target_ind,:].toarray())) # read input
            target_domain_labels = Variable(FloatTensor([1.,])) # set to 1

            if cuda:
                target_batch = target_batch.cuda()
                target_domain_labels = target_domain_labels.cuda()
            
            # Get the domain loss for the target instances:
            _, domain_out = model.forward(target_batch, alpha)
            target_domain_loss = domain_loss_fn(domain_out, target_domain_labels)

            # Get sum loss update weights:
            # domain adaptation:
            total_loss = task_loss + domain_weight* (domain_loss + target_domain_loss)
            # Task only:
            # total_loss = task_loss
            epoch_loss += total_loss
            total_loss.backward()
            optimizer.step()
        
        unique_source_inds = np.unique(selected_source_inds)
        all_source_inds = np.arange(num_train_instances)
        eval_source_inds = np.setdiff1d(all_source_inds, unique_source_inds)
        source_eval_X = X_train[eval_source_inds]
        source_eval_y = y_train[eval_source_inds]
        source_task_out, source_domain_out = model.forward( Variable(FloatTensor(source_eval_X.toarray())).cuda(), alpha=0.)
        # source domain is 0, count up predictions where 1 - prediction = 1
        source_domain_preds = np.sum(1 - source_domain_out.cpu().data.numpy())
        source_domain_acc = source_domain_preds / len(source_eval_y)

        source_y_pred = np.round(source_task_out.cpu().data.numpy()[:,0])
        # predictions of 1 are the positive class: tps are where prediction and gold are 1
        tps = np.sum(source_y_pred * source_eval_y)
        true_preds = source_y_pred.sum()
        true_labels = source_eval_y.sum()

        recall = tps / true_labels
        prec = 1 if tps == 0 else tps / true_preds
        f1 = 2 * recall * prec / (recall+prec)
        print("[Source] Epoch %d: loss=%f\tnum_insts=%d\tdom_acc=%f\tP=%f\tR=%f\tF=%f" % (epoch, epoch_loss, len(source_eval_y), source_domain_acc, prec, recall, f1))
        

        # No eval:
        # print("Epoch loss: %f" % (epoch_loss))

        # try an evaluation on the test data:
        # model.evaluate()

        ## To do an eval on the target set:        
        # target_data = Variable(FloatTensor(X_test.toarray())).cuda()
        # task_out, domain_out = model.forward(target_data, alpha=0.)

        # # Target domain is 1, predictions of 1 are predictions of target domain:
        # target_domain_preds = np.sum(domain_out.cpu().data.numpy())
        # domain_acc = target_domain_preds / num_test_instances

        # y_pred = np.round(task_out.cpu().data.numpy()[:,0])
        # tps = np.sum(y_pred * y_test)
        # true_preds = y_pred.sum()
        # true_labels = y_test.sum()

        # recall = tps / true_labels
        # prec = 1 if tps == 0 else tps / true_preds
        # f1 = 2 * recall * prec / (recall+prec)
        # print("[Target] Epoch %d: loss=%f\tdom_acc=%f\tP=%f\tR=%f\tF=%f" % (epoch, epoch_loss, domain_acc, prec, recall, f1))



if __name__ == "__main__":
    main(sys.argv[1:])