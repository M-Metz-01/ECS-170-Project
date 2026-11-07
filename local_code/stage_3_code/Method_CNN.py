
#imports
from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


# CNN model class; builds, trains, and predicts
# both course method and PyTorch model so Setting class can call run() on it and so PyTorch treats it as a
# trainable neural network
class Method_CNN(method, nn.Module):
    data = None

    # default parameters (subclasses and scripts override these)
    in_channels = 1                       #num of color channels in each image (1 = gray, 3 = RGB)
    image_size = (28, 28)                 #(height, width) of each image
    num_classes = 10                      #num of output classes
    conv_channels = [32, 64]              #list of output channels, one entry per conv block
    convs_per_block = 1                   #num of conv layers inside each block (before pooling)
    kernel_size = 3                       #size of the conv filters (kernel_size x kernel_size)
    padding = None                        #zero padding around the image (None = keep same size, kernel_size // 2)
    stride = 1                            #step size of the conv filters
    pooling = 'max'                       #max | avg | None  (pooling layer at the end of each block)
    pool_size = 2                         #size and stride of the pooling window
    hidden_sizes = [128]                  #list of fully connected hidden layer widths after the conv blocks
    activation = 'relu'                   #relu | tanh | sigmoid | leaky_relu | gelu
    batch_norm = False                    #var to add or remove batch normalization after each conv layer
    conv_dropout = 0.0                    #dropout probability after each conv block (0 = no dropout)
    dropout = 0.0                         #dropout probability in the fully connected layers
    augment = False                       #random horizontal flip + random crop on training batches
    max_epoch = 10                        #num of training epochs
    batch_size = 128                      #num of images per mini-batch
    eval_batch_size = 1000                #num of images per batch when predicting (keeps memory low)
    learning_rate = 1e-3                  #optimizer learning rate
    optimizer_name = 'adam'               # adam | adamw | sgd | sgd_momentum | rmsprop
    weight_decay = 0.0                    #weight decay passed to optimizer
    loss_name = 'cross_entropy'           #cross_entropy | label_smoothing | nll | mse
    lr_schedule = None                    # None | 'cosine' | 'step'
    device = None                         #None = use a cuda GPU if there is one, otherwise the cpu
    seed = 2                              # random seed
    history = None                        # filled in during training; holds loss/accuracy for each epoch

    def __init__(self, mName, mDescription, **kwargs):
        # creates the model: sets name, applies settings, and builds layers
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # overwrite default parameters with ones passed to it
        for k, v in kwargs.items():
            setattr(self, k, v)

        # pick the device to train on
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # set the seed before building so the starting weights are the same every run
        torch.manual_seed(self.seed)
        # build network layers
        self._build()
        self.to(self.device)

    def _act(self):
        # returns the activation function defined in the settings
        return {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
                'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}[self.activation]()

    def _build(self):
        # builds the conv layers, then the fully connected layers, from settings
        pad = self.kernel_size // 2 if self.padding is None else self.padding
        conv_layers, prev = [], self.in_channels

        # add one block for each entry in conv_channels: (conv -> [batch norm] -> activation) x convs_per_block -> pool
        for c in self.conv_channels:
            for _ in range(self.convs_per_block):
                conv_layers.append(nn.Conv2d(prev, c, kernel_size=self.kernel_size, stride=self.stride, padding=pad))
                if self.batch_norm:
                    conv_layers.append(nn.BatchNorm2d(c))           #batch norm if enabled
                conv_layers.append(self._act())
                prev = c
            if self.pooling == 'max':
                conv_layers.append(nn.MaxPool2d(self.pool_size))    #max pooling if enabled
            elif self.pooling == 'avg':
                conv_layers.append(nn.AvgPool2d(self.pool_size))    #average pooling if enabled
            if self.conv_dropout > 0:
                conv_layers.append(nn.Dropout2d(self.conv_dropout)) #dropout if enabled
        self.features = nn.Sequential(*conv_layers)

        # find out how many numbers come out of the conv blocks by pushing one blank image through them
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *self.image_size)
            prev = self.features(dummy).numel()

        # fully connected part: flatten -> hidden layers -> output layer (same as the stage 2 MLP)
        fc_layers = [nn.Flatten()]
        for h in self.hidden_sizes:
            fc_layers.append(nn.Linear(prev, h))           #linear layer
            fc_layers.append(self._act())
            if self.dropout > 0:
                fc_layers.append(nn.Dropout(self.dropout)) #dropout if enabled
            prev = h

        #add output layer: last hidden size -> one score per class
        fc_layers.append(nn.Linear(prev, self.num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        # send the images through the conv blocks, then the fully connected layers, and return scores
        return self.classifier(self.features(x))

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

    def _augment(self, xb):
        #randomly flips and shifts a batch of training images so the model sees slightly different images each epoch
        n, _, h, w = xb.shape
        #flip about half of the images left-to-right
        flip = torch.rand(n, device=xb.device) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        #pad by 4 pixels on every side, then cut a random h x w window back out of each image
        padded = nn.functional.pad(xb, (4, 4, 4, 4))
        dy = torch.randint(0, 9, (n,))
        dx = torch.randint(0, 9, (n,))
        return torch.stack([padded[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n)])

    def _predict_scores(self, X):
        #returns the class scores for every image in X, a few images at a time so memory stays small
        nn.Module.train(self, False)       #evaluation mode
        out = []
        with torch.no_grad():                    #no gradients needed
            for i in range(0, X.shape[0], self.eval_batch_size):
                out.append(self.forward(X[i:i + self.eval_batch_size].to(self.device)).cpu())
        return torch.cat(out)

    def train(self, X, y, X_val=None, y_val=None):
        #Trains model on X, y for max_epoch epochs and records results in history
        # note: nn.Module.train(mode) is overridden by the template's train(X, y);
        # we switch modes explicitly with nn.Module.train(self, ...) instead.

        #set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        #convert data to tensors (kept on the cpu; each batch is moved to the device when used)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long)
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
            total, correct = 0.0, 0
            #loop through mini-batches
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                #skip a final batch of 1 image if batch norm is on
                if len(idx) < 2 and self.batch_norm:
                    continue
                xb, yb = X[idx].to(self.device), y[idx].to(self.device)
                if self.augment:
                    xb = self._augment(xb)
                #predict on the batch and compute the loss
                logits = self.forward(xb)
                loss = self._loss(logits, yb)
                optimizer.zero_grad()       #reset gradients
                loss.backward()             #compute the gradients
                optimizer.step()            #update weights
                total += loss.item() * len(idx)      #add batch's loss to the epoch total
                correct += (logits.argmax(1) == yb).sum().item()  #count correct predictions in this batch

            #move learning-rate schedule forward one epoch
            if scheduler is not None:
                scheduler.step()

            # epoch-level monitoring
            # train loss/accuracy = running values over the mini-batches of this epoch
            # (re-predicting all 50,000 CIFAR / 60,000 MNIST training images every epoch would be too slow on a cpu)
            tr_acc = correct / n
            #default to nan if there is no extra set to measure
            te_loss = te_acc = float('nan')
            # loss and accuracy on the extra set if one was given
            if X_val is not None:
                lg = self._predict_scores(X_val)
                te_loss = self._loss(lg, y_val).item()
                accuracy_evaluator.data = {'true_y': y_val, 'pred_y': lg.argmax(1)}
                te_acc = accuracy_evaluator.evaluate()
            #save this epoch's results to history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(total / n)
            self.history['train_acc'].append(tr_acc)
            self.history['test_loss'].append(te_loss)
            self.history['test_acc'].append(te_acc)

            #print progress every epoch (CNN epochs are slow, so it's nice to see each one)
            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % (total / n), 'Train Acc: %.4f' % tr_acc,
                  'Test Loss: %.4f' % te_loss, 'Test Acc: %.4f' % te_acc)

    def test(self, X):
        #predicts the class for every image in X and returns the prediction
        scores = self._predict_scores(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        # pick the class with the highest score for each image
        return scores.argmax(1)

    def run(self):
        #Entry point called by the setting class: trains on the training data then predicts of the test data
        print('method running...')
        print('--start training...')
        #converts the test data to tensors
        Xte = torch.as_tensor(np.asarray(self.data['test']['X']), dtype=torch.float32)
        yte = torch.as_tensor(np.asarray(self.data['test']['y']), dtype=torch.long)
        # the test set is only *monitored* each epoch (for the learning curves);
        # it is never used to update weights.
        self.train(self.data['train']['X'], self.data['train']['y'], Xte, yte)
        print('--start testing...')
        #predict on the test images
        pred_y = self.test(self.data['test']['X'])
        #return predictions
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
