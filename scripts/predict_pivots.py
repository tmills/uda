#!/usr/bin/python
import sys
from os.path import join,exists,dirname

class PivotPredictorModel(nn.Module):

    def __init__(self, input_features, pivot_features, hidden_nodes=200):
        super(PivotPredictorModel, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('input_layer', nn.Linear(input_features, hidden_nodes))
        self.model.add_module(nn.Relu(True))
        self.model.add_module('output_layer', nn.Linear(hidden_nodes, pivot_features))
        ## TODO - how to add sigmoid to a bunch of independent outputs?
    
    def forward(self, input):
        return self.model(input)

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required arguments: <data file> <pivot inds file>\n")
        sys.exit(-1)
   
    cuda = False 
    if torch.cuda.is_available():
        cuda = True

    # Define constants and hyper-params    
    goal_ind = 2
    domain_weight = 0.1
    train_portion = 0.8
    lr = 0.01
    epochs = 100
    batch_size = 32

    # Read the data:
    sys.stderr.write("Reading source data from %s\n" % (args[0]))
    # Ignore y from files since we're using pivot features as y
    all_X, _ = load_svmlight_file(args[0])
    # Read pivot values from X:
    pivot_inds = read_pivot_file(args[1])
    all_Y = all_X[:, pivot_inds]
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
    train_X = all_X[train_inds,:]
    train_Y = all_Y[train_inds,:]
    dev_X = all_X[dev_inds,:]
    dev_Y = all_X[dev_inds,:]

    model = PivotPredictor(all_X.shape[0], len(pivot_inds))
    loss = # TODO

    if cuda:
        model.cuda()
        loss_fn.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for batch in range( 1+ ( num_instances // batch_size ) ):
            start_ind = batch * batch_size
            end_ind = num_instances-1 if start_ind + batch_size >= num_instances else start_ind+batch_size
            batch_X = train_X[start_ind:end_ind,:]
            batch_Y = train_Y[start_ind:end_ind,:]

            preds = model.forward(batch_X)
            loss = loss_fn(preds, batch_Y)
            epoch_loss += loss
            optimizer.step()



if __name__ == '__main__':
    main(sys.argv[1:])

