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
        # Totally random:
        # output = Variable(torch.randn(grad_output.shape).cuda()) + grad_output * 0 # grad_output.neg() * ctx.alpha
        # zero (ignores domain)
        # output = 0 * grad_output
        # reversed (default)
        # output = grad_output.neg() * ctx.alpha
        # print("Input grad is %s, output grad is %s" % (grad_output.data.cpu().numpy()[:10], output.data.cpu().numpy()[:10]))
        # return output, None
        return grad_output, None

# Instead of this, may be able to just regularize by forcing off-diagonal to zero
# didn't work bc/ of memory issues
# class StraightThroughLayer(nn.Module):
#     def __init__(self, input_features):
#         super(StraightThroughLayer, self).__init__()

#         self.vector = nn.Parameter( torch.ones(1, input_features) )
#         #self.add_module('pass-through vector', self.vector)

#     def forward(self,  input_data):
#         # output = input_data * self.vector
#         output = torch.mul(input_data, self.vector)
#         return output
        

class PivotLearnerModel(nn.Module):

    def __init__(self, input_features):
        super(PivotLearnerModel, self).__init__()
        # Feature takes you from input to the "representation"
        # self.feature = nn.Sequential()
        # straight through layer just does an element-wise product with a weight vector
        num_features = input_features
        self.vector = nn.Parameter( torch.randn(1, input_features) )
        # self.feature.add_module('input_layer', StraightThroughLayer(input_features))
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
        hidden_nodes = 100
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
        # feature = self.feature(input_data)
        feature = input_data * self.vector
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

    num_instances, num_feats = all_X.shape

    domain_map = read_feature_groups(join(dirname(args[0]), 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(dirname(args[0]), 'reduced-features-lookup.txt'))

    lr = 0.01
    epochs = 100

    direction = 1 if backward else 0
    sys.stderr.write("using domain %s as source, %s as target\n"  %
        (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

    train_instance_inds = np.where(all_X[:,domain_inds[direction]].toarray() > 0)[0]
    X_train = all_X[train_instance_inds,:]
    X_train[:, domain_inds[direction]] = 0
    X_train[:, domain_inds[1-direction]] = 0

    y_train = all_y[train_instance_inds]
    num_train_instances = X_train.shape[0]

    test_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
    X_test = all_X[test_instance_inds,:]
    X_test[:, domain_inds[direction]] = 0
    X_test[:, domain_inds[1-direction]] = 0
    # y_test = all_y[test_instance_inds]
    num_test_instances = X_test.shape[0]
    
    model = PivotLearnerModel(num_feats)
    task_loss_fn = nn.BCELoss()
    domain_loss_fn = nn.BCELoss()
    l1_loss = nn.L1Loss()
    

    if cuda:
        model.cuda()
        task_loss_fn.cuda()
        domain_loss_fn.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr)
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
            source_task_labels = Variable(torch.unsqueeze(FloatTensor([y_train[source_ind],]), 1))# read task labels
            source_domain_labels = Variable(torch.unsqueeze(FloatTensor([0.,]), 1)) # set to 0

            if cuda:
                source_batch = source_batch.cuda()
                source_task_labels = source_task_labels.cuda()
                source_domain_labels = source_domain_labels.cuda()
            
            # Get the task loss and domain loss for the source instance:
            task_out, task_domain_out = model.forward(source_batch, alpha)
            task_loss = task_loss_fn(task_out, source_task_labels)
            domain_loss = domain_loss_fn(task_domain_out, source_domain_labels)
            # domain2_loss = domain_loss_fn(task_domain_out[1], source_domain_labels)
            # reg_term = l1_loss(model.feature.input_layer.vector, torch.zeros_like(model.feature.input_layer.vector))

            # Randomly select a target instance:
            target_ind = randint(num_test_instances)
            target_batch = Variable(FloatTensor(X_test[target_ind,:].toarray())) # read input
            target_domain_labels = Variable(torch.unsqueeze(FloatTensor([1.,]), 1)) # set to 1

            if cuda:
                target_batch = target_batch.cuda()
                target_domain_labels = target_domain_labels.cuda()
            
            # Get the domain loss for the target instances:
            _, target_domain_out = model.forward(target_batch, alpha)
            target_domain_loss = domain_loss_fn(target_domain_out, target_domain_labels)
            # target_domain2_loss = domain_loss_fn(target_domain_out[1], target_domain_labels)

            # Get sum loss update weights:
            # domain adaptation:
            total_loss = task_loss + domain_weight * (domain_loss + target_domain_loss)
            # Task only:
            # total_loss = task_loss
            # Domain only:
            # total_loss = domain_loss + target_domain_loss
            # Debugging with 2 domain classifiers:
            # total_loss = domain_loss + domain2_loss + target_domain_loss + target_domain2_loss
            # With regularization and DA term:
            # total_loss = task_loss + domain_weight * (domain_loss + target_domain_loss) + reg_term
            # With regularization only:
            # total_loss = task_loss + reg_term

            epoch_loss += total_loss
            total_loss.backward()

            # for param in model.named_parameters():
            #     print(param[0])
            #     print(param[1])

            optimizer.step()

        print("Min weight: %f" % (model.vector.min()))
        print("Max weight: %f" % (model.vector.max()))
        print("Ave weight: %f" % (torch.abs(model.vector.sum())))

        # At the end of every epoch, examine domain accuracy and how many non-zero parameters we have
        unique_source_inds = np.unique(selected_source_inds)
        all_source_inds = np.arange(num_train_instances)
        eval_source_inds = np.setdiff1d(all_source_inds, unique_source_inds)
        source_eval_X = X_train[eval_source_inds]
        source_eval_y = y_train[eval_source_inds]
        source_task_out, source_domain_out = model.forward( Variable(FloatTensor(source_eval_X.toarray())).cuda(), alpha=0.)
        # source domain is 0, count up predictions where 1 - prediction = 1
        source_domain_preds = np.round(source_domain_out[0].cpu().data.numpy())
        source_predicted_count = np.sum(1 - source_domain_preds)
        source_domain_acc = source_predicted_count / len(source_eval_y)

        source_y_pred = np.round(source_task_out.cpu().data.numpy()[:,0])
        # predictions of 1 are the positive class: tps are where prediction and gold are 1
        tps = np.sum(source_y_pred * source_eval_y)
        true_preds = source_y_pred.sum()
        true_labels = source_eval_y.sum()

        recall = tps / true_labels
        prec = 1 if tps == 0 else tps / true_preds
        f1 = 2 * recall * prec / (recall+prec)

        # weights = model.feature[0].vector
        num_nonzero = 0 #(weights.data!=0).sum() 

        print("[Source] Epoch %d: loss=%f\tnnz=%d\tnum_insts=%d\tdom_acc=%f\tP=%f\tR=%f\tF=%f" % (epoch, epoch_loss, num_nonzero, len(source_eval_y), source_domain_acc, prec, recall, f1))

if __name__ == '__main__':
    main(sys.argv[1:])

