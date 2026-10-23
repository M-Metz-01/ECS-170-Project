'''
Configurable PyTorch MLP for multiclass classification.
'''

#imports
from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


# MLP model class; builds, trains, and predicts
# both course method and PyTorch model so Setting class can call run() on it and so PyTorch treats it as a
# trainable neural network
class Method_MLP(method, nn.Module):
    data = None

    # default parameters (scripts override these)
    input_dim = 784                       #num of input features (one per pixel)
    num_classes = 10                      #num of output classes (0-9)
    hidden_sizes = [128]                  #list of hidden layer widths
    activation = 'relu'                   #relu | tanh | sigmoid | leaky_relu | gelu
    dropout = 0.0                         #dropout probability (0 = no dropout)
    batch_norm = False                    #var to add or remove batch normalization to each hidden layer
    max_epoch = 50                        #num of training epochs
    batch_size = 128                      #num of images per mini-batch
    learning_rate = 1e-3                  #optimizer learning rate
    optimizer_name = 'adam'               # adam | adamw | sgd | sgd_momentum | rmsprop
    weight_decay = 0.0                    #weight decay passed to optimizer
    loss_name = 'cross_entropy'           #cross_entropy | label_smoothing | nll | mse
    lr_schedule = None                    # None | 'cosine' | 'step'
    seed = 2                              # random seed
    history = None                        # filled in during training; holds loss/accuracy for each epoch

    def __init__(self, mName, mDescription, **kwargs):
        # creates the model: sets name, applies settings, and builds layers
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # overwrite default parameters with ones passed to it
        for k, v in kwargs.items():
            setattr(self, k, v)

        # build network layers
        self._build()


    def _act(self):
        # returns the activation function defined in the settings
        return {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
                'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}[self.activation]()

    def _build(self):
        # builds network layers from settings
        layers, prev = [], self.input_dim

        # add one block for each hidden layer
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))              #linear layer
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(h))           #batch norm if enabled
            layers.append(self._act())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))    #dropout if enabled
            prev = h

        #add output layer: last hidden size = 10 class scores
        layers.append(nn.Linear(prev, self.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # send the input through all layers and return scores
        return self.net(x)

    def _optimizer(self):
        # creates the optimizer using chosen settings
        p, lr, wd = self.parameters(), self.learning_rate, self.weight_decay

        # pick optimizer named in settings
        return {
            'adam': lambda: torch.optim.Adam(p, lr=lr, weight_decay=wd),
            'adamw': lambda: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
            'sgd': lambda: torch.optim.SGD(p, lr=lr, weight_decay=wd),
            'sgd_momentum': lambda: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd),
            'rmsprop': lambda: torch.optim.RMSprop(p, lr=lr, weight_decay=wd),
        }[self.optimizer_name]()

    def _loss(self, logits, y):
        #computes the loss between the model's scores and the true labels using the chosen loss function
        if self.loss_name == 'cross_entropy':
            return nn.functional.cross_entropy(logits, y)
        if self.loss_name == 'label_smoothing':
            return nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        if self.loss_name == 'nll':
            return nn.functional.nll_loss(torch.log_softmax(logits, dim=1), y)
        if self.loss_name == 'mse':
            return nn.functional.mse_loss(torch.softmax(logits, dim=1),
                                          nn.functional.one_hot(y, self.num_classes).float())
        raise ValueError(self.loss_name)

    def train(self, X, y, X_val=None, y_val=None):
        #Trains model on X, y for max_epoch epochs and records results in history
        # note: nn.Module.train(mode) is overridden by the template's train(X, y);
        # we switch modes explicitly with self.training instead.

        #set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        #convert data to tensors
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))
        # create the optimizer
        optimizer = self._optimizer()
        # create the learning-rate scheduler if one was chosen
        scheduler = None
        if self.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        elif self.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, self.max_epoch // 3), gamma=0.3)
        #accuracy calculator
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        #empty history lists, filled in once per epoch
        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        n = X.shape[0]

        #loop through epochs
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            perm = torch.randperm(n)
            total = 0.0
            #loop through mini-batches
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                #skip a final batch of 1 image if batch norm is on
                if len(idx) < 2 and self.batch_norm:
                    continue
                #predict on the batch and compute the loss
                loss = self._loss(self.forward(X[idx]), y[idx])
                optimizer.zero_grad()       #reset gradients
                loss.backward()             #compute the gradients
                optimizer.step()            #update weights
                total += loss.item() * len(idx)      #add batch's loss to the epoch total

            #move learning-rate schedule forward one epoch
            if scheduler is not None:
                scheduler.step()

            # epoch-level monitoring (train loss = average over minibatches during the epoch)
            nn.Module.train(self, False)     #switch to evalulation mode
            with torch.no_grad():                  #no gradients while measuring
                # training accuracy on full training set
                tr_pred = self.forward(X).argmax(1)
                accuracy_evaluator.data = {'true_y': y, 'pred_y': tr_pred}
                tr_acc = accuracy_evaluator.evaluate()
                #default to nan if there is no extra set to measure
                te_loss = te_acc = float('nan')
                # loss and accuracy on the extra set if one was given
                if X_val is not None:
                    lg = self.forward(X_val)
                    te_loss = self._loss(lg, y_val).item()
                    accuracy_evaluator.data = {'true_y': y_val, 'pred_y': lg.argmax(1)}
                    te_acc = accuracy_evaluator.evaluate()
            #save this epoch's results to history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(total / n)
            self.history['train_acc'].append(tr_acc)
            self.history['test_loss'].append(te_loss)
            self.history['test_acc'].append(te_acc)

            #print progress every 5 epochs and at last for testing
            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % (total / n), 'Train Acc: %.4f' % tr_acc,
                      'Test Acc: %.4f' % te_acc)

    def test(self, X):
        #predicts the digit for every image in X and returns the prediction
        nn.Module.train(self, False)       #evalulation mode
        with torch.no_grad():                    #no gradients needed
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # pick the class wit hthe highest score for each image
        return y_pred.argmax(1)

    def run(self):
        #Entry point called by the setting class: trains on the training data then predicts of the test data
        print('method running...')
        print('--start training...')
        #converts the test data to tensors
        Xte = torch.FloatTensor(np.array(self.data['test']['X']))
        yte = torch.LongTensor(np.array(self.data['test']['y']))
        # the test set is only *monitored* each epoch (for the convergence plot);
        # it is never used to update weights.
        self.train(self.data['train']['X'], self.data['train']['y'], Xte, yte)
        print('--start testing...')
        #predict on the test images
        pred_y = self.test(self.data['test']['X'])
        #return predictions
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
