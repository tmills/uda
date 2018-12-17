#!/usr/bin/python
import sys
from os.path import join,exists,dirname
import random

import numpy as np
from numpy.random import randint, choice
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
        # Totally random:
        # output = Variable(torch.randn(grad_output.shape).cuda()) + grad_output * 0 # grad_output.neg() * ctx.alpha
        # zero (ignores domain)
        # output = 0 * grad_output
        # reversed (default)
        output = grad_output.neg() * ctx.alpha
        # print("Input grad is %s, output grad is %s" % (grad_output.data.cpu().numpy()[:10], output.data.cpu().numpy()[:10]))
        return output, None

# Instead of this, may be able to just regularize by forcing off-diagonal to zero
# didn't work bc/ of memory issues
class StraightThroughLayer(nn.Module):
    def __init__(self, input_features):
        super(StraightThroughLayer, self).__init__()

        self.vector = nn.Parameter( torch.randn(1, input_features) )
        #self.add_module('pass-through vector', self.vector)

    def forward(self,  input_data):
        # output = input_data * self.vector
        output = torch.mul(input_data, self.vector)
        return output
        

class PivotLearnerModel(nn.Module):

    def __init__(self, input_features):
        super(PivotLearnerModel, self).__init__()
        # Feature takes you from input to the "representation"
        # self.feature = nn.Sequential()
        # straight through layer just does an element-wise product with a weight vector
        num_features = input_features
        # num_features = 200
        # self.vector = nn.Parameter( torch.randn(1, input_features) )
        self.feature = nn.Sequential() 
        self.feature.add_module('input_layer', StraightThroughLayer(input_features))
        # self.feature.add_module('feature_layer', nn.Linear(input_features, num_features))
        self.feature.add_module('relu', nn.ReLU(True))

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

        self.domain_classifier.add_module('domain_classifier', nn.Linear(num_features, 1, bias=False))
        # # self.domain_classifier.add_module('domain_predict', nn.Linear(100, 1))
        self.domain_classifier.add_module('domain_sigmoid', nn.Sigmoid())

        # self.domain_classifier2 = nn.Sequential()
        # self.domain_classifier2.add_module('domain_linear', nn.Linear(num_features, 1, bias=False))
        # # # self.domain_classifier.add_module('domain_predict', nn.Linear(100, 1))
        # self.domain_classifier2.add_module('domain_sigmoid', nn.Sigmoid())

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        # feature = input_data * self.vector
        task_prediction = self.task_classifier(feature)

        # Get domain prediction
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_prediction = self.domain_classifier(reverse_feature)
        # Only domain predictor 1 is reversed
        # domain_prediction2 = self.domain_classifier2(feature)

        return task_prediction, domain_prediction #(domain_prediction, domain_prediction2)


def main(args):
    if len(args) < 1:
        sys.stderr.write("Required arguments: <data file> [backward True|False]\n")
        sys.exit(-1)
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    if len(args) > 1:
        backward = bool(args[1])
        print("Direction is backward based on args=%s" % (args[1]))
    else:
        backward = False
        print("Direction is forward by default")

    # Read the data:
    goal_ind = 2
    domain_weight = 1.0
    reg_weight = 0.1
    lr = 0.01
    epochs = 1000
    batch_size = 50

    sys.stderr.write("Reading source data from %s\n" % (args[0]))
    all_X, all_y = load_svmlight_file(args[0])
    
    # y is 1,2 by default, map to 0,1 for sigmoid training
    all_y -= 1   # 0/1
    # continue to -1/1 for softmargin training:
    # all_y *= 2  # 0/2
    # all_y -= 1  # -1/1

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
    # y_target_train = y_target[:num_target_train]
    X_target_valid = X_target[num_target_train:, :]
    # y_target_dev = y_target[num_target_train:]

    # y_test = all_y[target_instance_inds]
    num_target_instances = X_target_train.shape[0]
    
    model = PivotLearnerModel(num_feats).to(device)
    task_loss_fn = nn.BCELoss()
    domain_loss_fn = nn.BCELoss()
    l1_loss = nn.L1Loss()
    
    #task_loss_fn.cuda()
    #    domain_loss_fn.cuda()
    #    l1_loss.cuda()

    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=lr)

    # weights = model.vector
    try:
        weights = model.feature.input_layer.vector
        print("Before training:")
        print("Min (abs) weight: %f" % (torch.abs(weights).min()))
        print("Max (abs) weight: %f" % (torch.abs(weights).max()))
        print("Ave weight: %f" % (torch.abs(weights).mean()))
        num_zeros = (weights.data==0).sum() 
        near_zeros = (torch.abs(weights.data)<0.000001).sum()
        print("Zeros=%d, near-zeros=%d" % (num_zeros, near_zeros))
    except:
        pass

    # Main training loop
    inds = np.arange(num_train_instances)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        # Do a training epoch:
        for batch in range( 1+ ( num_train_instances // batch_size ) ):
            model.zero_grad()

            start_ind = batch * batch_size
            if start_ind >= num_train_instances:
                #This happens if our number of instances is perfectly divisible by batch size (when batch_size=1 this is often).
                break
            end_ind = num_train_instances if start_ind + batch_size >= num_train_instances else start_ind+batch_size
            this_batch_size = end_ind - start_ind

            ## Gradually increase (?) the importance of the regularization term
            ave_ind = start_ind + this_batch_size // 2
            p = float(ave_ind + epoch * num_train_instances*2) / (epochs * num_train_instances*2)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_batch = FloatTensor(X_task_train[start_ind:end_ind,:].toarray()).to(device) # read input
            source_task_labels = torch.unsqueeze(FloatTensor([y_task_train[start_ind:end_ind],]).to(device), 1)# read task labels
            source_domain_labels = torch.zeros(this_batch_size,1, device=device) # set to 0
           
            # Get the task loss and domain loss for the source instance:
            task_out, task_domain_out = model.forward(source_batch, alpha)
            task_loss = task_loss_fn(task_out, source_task_labels)
            domain_loss = domain_loss_fn(task_domain_out, source_domain_labels)
            # domain2_loss = domain_loss_fn(task_domain_out[1], source_domain_labels)
            try:
                weights = model.feature.input_layer.vector
                reg_term = l1_loss(weights, torch.zeros_like(weights, device=device))
            except:
                reg_term = 0

            # Randomly select a matching number of target instances:
            target_inds = choice(num_target_instances, this_batch_size, replace=False)
            target_batch = FloatTensor(X_target_train[target_inds,:].toarray()).to(device) # read input
            target_domain_labels = torch.ones(this_batch_size, 1, device=device)

           
            # Get the domain loss for the target instances:
            _, target_domain_out = model.forward(target_batch, alpha)
            target_domain_loss = domain_loss_fn(target_domain_out, target_domain_labels)
            # target_domain2_loss = domain_loss_fn(target_domain_out[1], target_domain_labels)

            # Get sum loss update weights:
            # domain adaptation:
            # total_loss = task_loss + domain_weight * (domain_loss + target_domain_loss)
            # Task only:
            # total_loss = task_loss
            # Domain only:
            # total_loss = domain_loss + target_domain_loss
            # Debugging with 2 domain classifiers:
            # total_loss = domain_loss + domain2_loss + target_domain_loss + target_domain2_loss
            # With regularization and DA term:
            total_loss = (task_loss + 
                            domain_weight * (domain_loss + target_domain_loss) + 
                            reg_weight * reg_term)
            # With regularization only:
            # total_loss = task_loss + reg_term

            epoch_loss += total_loss
            total_loss.backward()

            # for param in model.named_parameters():
            #     print(param[0])
            #     print(param[1])

            optimizer.step()


        # At the end of every epoch, examine domain accuracy and how many non-zero parameters we have
        # unique_source_inds = np.unique(selected_source_inds)
        # all_source_inds = np.arange(num_train_instances)
        # eval_source_inds = np.setdiff1d(all_source_inds, unique_source_inds)
        # source_eval_X = X_train[eval_source_inds]
        # source_eval_y = y_train[eval_source_inds]
        source_eval_X = X_task_valid
        source_eval_y = y_task_valid
        source_task_out, source_domain_out = model.forward( FloatTensor(source_eval_X.toarray()).to(device), alpha=0.)
        # If using BCEWithLogitsLoss which would automatically do a sigmoid post-process
        # source_task_out = nn.functional.sigmoid(source_task_out)
        # source_domain_out = nn.functional.sigmoid(source_domain_out)

        # source domain is 0, count up predictions where 1 - prediction = 1
        # If using sigmoid outputs (0/1) with BCELoss
        source_domain_preds = np.round(source_domain_out.cpu().data.numpy())
        # if using Softmargin() loss (-1/1) with -1 as source domain
        # source_domain_preds = np.round(((source_domain_out.cpu().data.numpy() * -1) + 1) / 2)
        source_predicted_count = np.sum(1 - source_domain_preds)
        source_domain_acc = source_predicted_count / len(source_eval_y)

        target_eval_X = X_target_valid
        _, target_domain_out = model.forward( FloatTensor(target_eval_X.toarray()).to(device), alpha=0.)
        # If ussing with BCEWithLogitsLoss (see above)
        # target_domain_out = nn.functional.sigmoid(target_domain_out)
        # if using sigmoid output (0/1) with BCELoss
        target_domain_preds = np.round(target_domain_out.cpu().data.numpy())
        # if using Softmargin loss (-1/1) with 1 as target domain:
        # target_domain_preds = np.round(((source_domain_out.cpu().data.numpy()) + 1) / 2)
        target_predicted_count = np.sum(target_domain_preds)

        domain_acc = (source_predicted_count + target_predicted_count) / (source_eval_X.shape[0] + target_eval_X.shape[0])

        # if using 0/1 predictions:
        source_y_pred = np.round(source_task_out.cpu().data.numpy()[:,0])
        # if using -1/1 predictions? (-1 = not negated, 1 = negated)
        # source_y_pred = np.round((source_task_out.cpu().data.numpy()[:,0] + 1) / 2)
        # source_eval_y += 1
        # source_eval_y /= 2

        # predictions of 1 are the positive class: tps are where prediction and gold are 1
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

        print("[Source] Epoch %d: loss=%f\tzeros=%d\tnear_zeros=%d\tnum_insts=%d\tdom_acc=%f\tP=%f\tR=%f\tF=%f" % (epoch, epoch_loss, num_zeros, near_zeros, len(source_eval_y), domain_acc, prec, recall, f1))

    weights = model.feature.input_layer.vector
    ranked_inds = torch.sort(torch.abs(weights))[1]
    pivots = ranked_inds[0,-1000:]
    pivot_list = pivots.cpu().data.numpy().tolist()
#    pivot_list.sort()
    for pivot in pivot_list:
        print('%d : %s' % (pivot, feature_map[pivot]))


if __name__ == '__main__':
    main(sys.argv[1:])

