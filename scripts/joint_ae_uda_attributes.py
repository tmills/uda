#!/usr/bin/env python3
import sys
import os
from os.path import join,exists,dirname
import random
from datetime import datetime
import pickle
import time
import argparse


import torch.nn as nn
import torch.optim as optim
import torch
from torch import sigmoid
from torch.nn.functional import relu
from torch.autograd import Function
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

from uda_common import read_feature_groups, read_feature_lookup

class JointLearnerModel(nn.Module):

    def __init__(self, num_features, num_pivots, pivot_hidden_nodes=100, dropout=0.5):
        super(JointLearnerModel, self).__init__()

        # The task net takes a concatenated input vector + predicted pivot vector and maps it to a prediction for the task
        self.task_net = nn.Sequential()
        
        # task_classifier maps from a feature representation to a task prediction
        self.task_classifier = nn.Linear(pivot_hidden_nodes,1)
        
        # domain classifier maps from a feature representation to a domain prediction
        #self.pivot_ae = nn.Sequential()
        self.rep_projector = nn.Linear(num_features, pivot_hidden_nodes)
        self.rep_predictor = nn.Linear(pivot_hidden_nodes, num_pivots)

        
    def forward(self, full_input):

        # Get predictions for all pivot candidates:
        pivot_rep = sigmoid(self.rep_projector(full_input))

        # Get pivot prediction
        pivot_pred = self.rep_predictor(pivot_rep)

        # Get task prediction
        task_prediction = self.task_classifier( pivot_rep )

        return task_prediction, pivot_pred, pivot_rep

def get_shuffled(X_feats, X_ae=None, y=None):
    inds = np.arange(X_feats.shape[0])
    np.random.shuffle(inds)
    shuffled_X_feats = X_feats[inds, :]

    if X_ae is None:
        shuffled_X_ae = None
    else:
        shuffled_X_ae = X_ae[inds, :]

    if y is None:
        shuffled_y = None
    else:
        shuffled_y = y[inds]
    return shuffled_X_feats, shuffled_X_ae, shuffled_y, inds

def GetTopNMI(n,X,labels):
    MI = []
    length = X.shape[1]


    for i in range(length):
        temp=mutual_info_score(X[:, i], labels)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI

def log(msg):
    sys.stdout.write('%s\n' % msg)
    sys.stdout.flush()

def train_model(X_source_ae, y_source, X_target_ae, ae_input_inds, ae_output_inds, y_target=None, X_unlabeled_ae=None, y_unlabeled_dom=None, X_source_valid_ae=None, y_source_valid=None):
    assert X_source_ae.shape[1] == X_target_ae.shape[1], "Source and target training data do not have the same number of features!"
    assert (X_unlabeled_ae is None) or (X_source_ae.shape[1] == X_unlabeled_ae.shape[1]), "Source and unlabeled training data do not have the same number of features!"

    device='cuda' if torch.cuda.is_available() else 'cpu'
 
    epochs = 50
    recon_weight = 100.0
    l2_weight = 0.1 #1
    # oracle_weight = 1.0
    max_batch_size = 50
    pivot_hidden_nodes = 2000
    weight_decay = 0.0000 #1
    lr = 0.001  # adam default is 0.001
    dropout=0.0
 
    log('Proceeding in standard semi-supervised pivot-learning mode')
    
    log('There are %d auto-encoder output features that meet source and target frequency requirements and %d predictors' % (len(ae_output_inds), len(ae_input_inds)))
    
    num_source_instances, num_features = X_source_ae.shape
    num_target_instances = X_target_ae.shape[0]
    if num_source_instances > num_target_instances:
        source_batch_size = max_batch_size
        num_batches = (num_source_instances // max_batch_size)
        target_batch_size = (num_target_instances // num_batches) 
    else:
        target_batch_size = max_batch_size
        num_batches = (num_target_instances // max_batch_size)
        source_batch_size = (num_source_instances // num_batches)

    num_unlabeled_instances = 0
    un_batch_size = 0
    if not X_unlabeled_ae is None:
        num_unlabeled_instances = X_unlabeled_ae.shape[0]
        if num_unlabeled_instances > num_source_instances:
            un_batch_size = num_unlabeled_instances // num_batches
            log("Unlabeled data will be processed in batches of size %d" % (un_batch_size))
        else:
            raise Exception("ERROR: There are too few unlabeled instances. Is something wrong?\n")

    model = JointLearnerModel(len(ae_input_inds), len(ae_output_inds), pivot_hidden_nodes=pivot_hidden_nodes, dropout=dropout).to(device)
    task_lossfn = nn.BCEWithLogitsLoss().to(device)
    recon_lossfn = nn.BCEWithLogitsLoss().to(device)
    l2_lossfn = nn.MSELoss(reduction='sum').to(device)

    opt = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=5, factor=0.33)
    best_valid_acc = 0
    best_valid_loss = -1

    try:
        for epoch in range(epochs):
            source_batch_ind = 0
            target_batch_ind = 0
            un_batch_ind = 0
            epoch_loss = 0
            tps = 0             # num for P/R
            true_labels = 0     # denom for recall
            true_preds = 0      # denom for prec
            correct_preds = 0   # For accuracy

            source_X_ae, _, source_y,_ = get_shuffled(X_source_ae, None, y_source)
            target_X_ae, _, target_y,_ = get_shuffled(X_target_ae, None, y_target)
            if not X_unlabeled_ae is None:
                unlabeled_X, unlabeled_X_ae,_,unlabeled_inds = get_shuffled(X_unlabeled_ae)
                batch_unlabeled_dom_y = y_unlabeled_dom[unlabeled_inds]

            model.train()
            for batch in range(num_batches):
                model.zero_grad()
                opt.zero_grad()

                ae_inputs = torch.FloatTensor(source_X_ae[source_batch_ind:source_batch_ind+source_batch_size,ae_input_inds].toarray()).to(device)

                task_pred,pivot_pred,_ = model(ae_inputs)

                # Get task loss:
                batch_source_y = torch.FloatTensor(source_y[source_batch_ind:source_batch_ind+source_batch_size]).to(device).unsqueeze(1)
                task_loss = task_lossfn(task_pred, batch_source_y)

                # Get reconstruction loss:
                ae_outputs = torch.FloatTensor(source_X_ae[source_batch_ind:source_batch_ind+source_batch_size,ae_output_inds].toarray()).to(device)
                source_recon_loss = recon_lossfn(pivot_pred, ae_outputs)
                
                # since we don't have a sigmoid in our network's task output (it is part of the loss function for numerical stability), we need to manually apply the sigmoid if we want to do some standard acc/p/r/f calculations.
                task_bin_pred = np.round(sigmoid(task_pred).data.cpu().numpy())[:,0]
                true_preds += task_bin_pred.sum().item()
                true_labels += batch_source_y.sum().item()
                tps += (task_bin_pred * batch_source_y[:,0]).sum().item()
                correct_preds += (task_bin_pred == batch_source_y[:,0]).sum().item()


                # pass it target examples and compute reconstruction loss:
                ae_inputs = torch.FloatTensor(target_X_ae[target_batch_ind:target_batch_ind+target_batch_size,ae_input_inds].toarray()).to(device)

                target_task_pred,pivot_pred,_ = model(ae_inputs)

                # No task loss because we don't have target labels

                # Reconstruction loss:
                ae_outputs = torch.FloatTensor(target_X_ae[target_batch_ind:target_batch_ind+target_batch_size,ae_output_inds].toarray()).to(device)
                target_recon_loss = recon_lossfn(pivot_pred, ae_outputs)

                # do representation learning on the unlabeled instances
                unlabeled_recon_loss = 0
                if num_unlabeled_instances > 0:
                    num_sub_batches = 1 + (un_batch_size // max_batch_size)
                    sub_batch_start_ind = 0
                    unlabeled_recon_loss = 0.0
                    for sub_batch in range(num_sub_batches):
                        sub_batch_size = min(max_batch_size, un_batch_size - sub_batch*max_batch_size )
                        if sub_batch_size <= 0:
                            log('Found an edge case where sub_batch_size<=0 with un_batch_size=%d' % (un_batch_size))
                            break

                        ae_inputs = torch.FloatTensor(unlabeled_X_ae[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size,ae_input_inds].toarray()).to(device)
                        ae_outputs = torch.FloatTensor(unlabeled_X_ae[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size, ae_output_inds].toarray()).to(device)
                        _, pivot_pred,_ = model(ae_inputs)
                        sub_batch_dom_y = torch.FloatTensor( batch_unlabeled_dom_y[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size]).to(device)
                        unlabeled_recon_loss += recon_lossfn(pivot_pred, ae_outputs)
                        sub_batch_start_ind += max_batch_size

                l2_loss = l2_lossfn( model.task_classifier.weight, torch.zeros_like(model.task_classifier.weight))
                # Compute the total loss and step the optimizer in the right direction:
                if batch == 0:
                    log('Epoch %d: task=%f l2=%f src_recon=%f, tgt_recon=%f, un_recon=%f' % 
                        (epoch, task_loss, l2_loss, source_recon_loss, target_recon_loss, unlabeled_recon_loss))
                total_loss = (task_loss + 
                                l2_weight * l2_loss +
                            #  t2_weight * task2_loss +
                            #  dom_weight * dom_loss +
                                recon_weight * (source_recon_loss + target_recon_loss + unlabeled_recon_loss))
                epoch_loss += total_loss.item()
                total_loss.backward()
            
                opt.step()
                
                source_batch_ind += source_batch_size
                target_batch_ind += target_batch_size
                un_batch_ind += un_batch_size

            # Print out some useful info: losses, weights of pivot filter, accuracy
            acc = correct_preds / source_batch_ind

            log("Epoch %d finished: loss=%f" % (epoch, epoch_loss) )
            log("  Training accuracy=%f" % (acc))

            model.eval()
            if not X_source_valid_ae is None:
                test_X = torch.FloatTensor(X_source_valid_ae[:, ae_input_inds].toarray()).to(device)
                raw_preds = model(test_X)[0]
                y_source_valid_pt = torch.FloatTensor(y_source_valid).to(device).unsqueeze(1)
                valid_task_loss = task_lossfn(raw_preds, y_source_valid_pt)
                test_preds = np.round(sigmoid(raw_preds).data.cpu().numpy())[:,0]
                correct_preds = (y_source_valid == test_preds).sum()
                true_pos_preds = (y_source_valid * test_preds).sum()
                
                acc = correct_preds / len(y_source_valid)
                prec = true_pos_preds / test_preds.sum()
                rec = true_pos_preds / y_source_valid.sum()
                if np.isnan(prec):
                    # if prec is nan, that means we didn't make any positive predictions, so recall=0.0, so
                    # making p=1 doesn't inflate f1.
                    prec = 1.0
                f1 = 2 * prec * rec / (prec + rec)

                log("  Validation loss=%f, accuracy=%f, p/r/f=%f/%f/%f" % (valid_task_loss, acc, prec, rec, f1))
                # sched.step(acc)
                new_lr = [ group['lr'] for group in opt.param_groups ][0]
                if new_lr != lr:
                    log("Learning rate modified to %f" % (new_lr))
                    lr = new_lr

                #if acc > best_valid_acc:
                if valid_task_loss < best_valid_loss or best_valid_loss < 0:
                    sys.stderr.write('Writing model at epoch %d\n' % (epoch))
                    sys.stderr.flush()
                    #best_valid_acc = acc
                    best_valid_loss = valid_task_loss 
                    torch.save(model, 'best_model.pt')

            if num_unlabeled_instances > 0:
                del unlabeled_X, unlabeled_X_ae
    except KeyboardInterrupt:
        log('Exiting training early due to keyboard interrupt')

    print("best validation loss/accuracy during training: %f %f" % (best_valid_loss, best_valid_acc))
    return best_valid_loss

def get_ae_inds(method, X_source_ae, X_target_ae, X_unlabeled_ae, train_labels, num_pivots, featnames, pivot_min_count=10):
    if method.startswith('freq'):
        source_cands = np.where(X_source_ae.sum(0) > pivot_min_count)[1]
        target_cands = np.where(X_target_ae.sum(0) > pivot_min_count)[1]
        # pivot candidates are those that meet frequency cutoff in both domains train data:
        ae_output_inds = np.intersect1d(source_cands, target_cands)

        if method == 'freq':
            # non-pivot candidates are the set difference - those that didn't meet the frequency cutoff in both domains:
            ae_input_inds = np.setdiff1d(range(X_source_ae.shape[1]), ae_output_inds)
        elif method == 'freq-ae':
            ae_input_inds = range(X_unlabeled_ae.shape[1])
    elif method == 'ae':
        ae_input_inds = ae_output_inds = range(X_source_ae.shape[1])
    elif method.startswith('mi'):
        # Run the sklearn mi feature selection:
        MIs, MI = GetTopNMI(2000, X_source_ae.toarray(), train_labels)
        MIs.reverse()
        ae_output_inds = []
        i=c=0
        while c < num_pivots:
            s_count = X_source_ae[:,MIs[i]].sum()
            t_count = X_target_ae[:,MIs[i]].sum()
            if s_count >= pivot_min_count and t_count >= pivot_min_count:
                ae_output_inds.append(MIs[i])
                c += 1
                print("feature %d is '%s' with mi %f" % (c, featnames[MIs[i]], MI[MIs[i]]))
            i += 1

        ae_output_inds.sort()
        if method == 'mi':
            ae_input_inds = np.setdiff1d(range(X_unlabeled_ae.shape[1]), ae_output_inds)
        elif method == 'mi-ae':
            ae_input_inds = range(X_source_ae.shape[1])
    return ae_input_inds, ae_output_inds


parser = argparse.ArgumentParser(description='PyTorch joint domain adaptation neural network trainer')
parser.add_argument('model_dir', nargs=1, help='Directory with model files')
parser.add_argument('-b', '--backward', action="store_true")
parser.add_argument('-m', '--method', default='mi-ae', choices=['freq', 'mi', 'ae', 'mi-ae', 'freq-ae'])

def main(args):
    if not torch.cuda.is_available():
        sys.stderr.write('WARNING: CUDA is not available... this may run very slowly!')

    args = parser.parse_args()

    dev_ratio = 0.15
    model_dir = args.model_dir[0]
    all_X, all_y = load_svmlight_file(join(model_dir, 'training-data_reduced.liblinear0'))
    all_y -= 1   # 0/1

    num_instances, num_feats = all_X.shape
    domain_map = read_feature_groups(join(dirname(model_dir), 'reduced-feature-groups.txt'))
    domain_inds = domain_map['Domain']

    feature_map = read_feature_lookup(join(dirname(model_dir), 'reduced-features-lookup.txt'))
    direction = 1 if args.backward else 0

    log("using domain %s as source, %s as target\n"  %
        (feature_map[domain_inds[direction]],feature_map[domain_inds[1-direction]]))

    source_instance_inds = np.where(all_X[:,domain_inds[direction]].toarray() > 0)[0]
    X_source = all_X[source_instance_inds,:]
    source_size = X_source.shape[0]
    X_source[:, domain_inds[direction]] = 0
    X_source[:, domain_inds[1-direction]] = 0
    y_source = all_y[source_instance_inds]
    train_size = int(source_size * (1 - dev_ratio))
    train_inds = np.random.choice(source_size, train_size, replace=False)
    dev_inds = np.setdiff1d(np.arange(source_size), train_inds)
    X_source_train = X_source[train_inds,:]
    y_source_train = y_source[train_inds]
    X_source_dev = X_source[dev_inds,:]
    y_source_dev = y_source[dev_inds]

    target_instance_inds = np.where(all_X[:,domain_inds[1-direction]].toarray() > 0)[0]
    X_target = all_X[target_instance_inds,:]
    X_target[:, domain_inds[direction]] = 0
    X_target[:, domain_inds[1-direction]] = 0
    y_target = all_y[target_instance_inds]

    num_target_train = int(X_target.shape[0] * 0.8)
    X_target_train = X_target[:num_target_train,:]
    # y_target_train = y_target[:num_target_train]
    X_target_valid = X_target[num_target_train:, :]
    # y_target_dev = y_target[num_target_train:]

    un_count = 40
    num_pivots = 100   # Only used in configuration that uses MI to get pivots
    pivot_min_count = 10

    lbl_num = 1000

    unlabeled=None

    ae_input_inds, ae_output_inds = get_ae_inds(args.method, 
                            X_source_train, 
                            X_target, 
                            None, 
                            y_source_train, 
                            num_pivots, 
                            feature_map,
                            pivot_min_count)
    
    loss = train_model(X_source_train,
                y_source_train,
                X_target,
                ae_input_inds,
                ae_output_inds,
                X_unlabeled_ae=None,
                X_source_valid_ae=X_source_dev,
                y_source_valid=y_source_dev)

    if loss <= 0:
        log('Cannot evaluate because no good model was saved.')
        return

    best_model = torch.load('best_model.pt')
    best_model.eval()

    device='cuda' if torch.cuda.is_available() else 'cpu'

    correct_preds = 0 
    tps = 0
    test_true_preds = 0
    for ind in range(X_target.shape[0]):
        target_X = torch.FloatTensor(X_target[ind,ae_input_inds].toarray()).to(device)
        target_test_predict_raw,_,_ = best_model(target_X)
        target_test_predict = np.round(sigmoid(target_test_predict_raw).data.cpu().numpy())[:,0]
        tps += (target_test_predict * y_target[ind])
        test_true_preds += target_test_predict
        correct_preds += (y_target[ind] == target_test_predict)
    acc = correct_preds / len(y_target)

    #tps = (target_test_predict * y_target).sum().item()
    test_true_labels = y_target.sum()
    #test_true_preds = target_test_predict.sum()
    rec = tps / test_true_labels
    prec = tps / test_true_preds
    f1 = 2 * rec * prec / (rec + prec)
    log("Target accuracy=%f, p/r/f1=%f/%f/%f" % (acc, prec, rec, f1))



if __name__ == '__main__':
    main(sys.argv[1:])

