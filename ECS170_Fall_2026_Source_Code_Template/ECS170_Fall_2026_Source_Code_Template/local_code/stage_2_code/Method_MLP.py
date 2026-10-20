'''
Configurable PyTorch MLP for multiclass classification.
'''

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # ---- hyper-parameters (override from the script) ----
    input_dim = 784
    num_classes = 10
    hidden_sizes = [128]          # list of hidden layer widths
    activation = 'relu'           # relu | tanh | sigmoid | leaky_relu | gelu
    dropout = 0.0
    batch_norm = False
    max_epoch = 50
    batch_size = 128
    learning_rate = 1e-3
    optimizer_name = 'adam'       # adam | adamw | sgd | sgd_momentum | rmsprop
    weight_decay = 0.0
    loss_name = 'cross_entropy'   # cross_entropy | label_smoothing | nll | mse
    lr_schedule = None            # None | 'cosine' | 'step'
    seed = 2
    # ---- filled in during training (for convergence plots) ----
    history = None

    def __init__(self, mName, mDescription, **kwargs):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._build()

    def _act(self):
        return {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
                'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}[self.activation]()

    def _build(self):
        layers, prev = [], self.input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(self._act())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, self.num_classes))
        # The network outputs raw scores (logits). The softmax is folded into the loss
        # (nn.CrossEntropyLoss applies log-softmax internally), which is the numerically
        # stable form; applying nn.Softmax before CrossEntropyLoss would softmax twice.
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def _optimizer(self):
        p, lr, wd = self.parameters(), self.learning_rate, self.weight_decay
        return {
            'adam': lambda: torch.optim.Adam(p, lr=lr, weight_decay=wd),
            'adamw': lambda: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
            'sgd': lambda: torch.optim.SGD(p, lr=lr, weight_decay=wd),
            'sgd_momentum': lambda: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd),
            'rmsprop': lambda: torch.optim.RMSprop(p, lr=lr, weight_decay=wd),
        }[self.optimizer_name]()

    def _loss(self, logits, y):
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
        # note: nn.Module.train(mode) is overridden by the template's train(X, y);
        # we switch modes explicitly with self.training instead.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))
        optimizer = self._optimizer()
        scheduler = None
        if self.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        elif self.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, self.max_epoch // 3), gamma=0.3)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        n = X.shape[0]
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2 and self.batch_norm:
                    continue
                loss = self._loss(self.forward(X[idx]), y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)
            if scheduler is not None:
                scheduler.step()

            # epoch-level monitoring (train loss = average over minibatches during the epoch)
            nn.Module.train(self, False)
            with torch.no_grad():
                tr_pred = self.forward(X).argmax(1)
                accuracy_evaluator.data = {'true_y': y, 'pred_y': tr_pred}
                tr_acc = accuracy_evaluator.evaluate()
                te_loss = te_acc = float('nan')
                if X_val is not None:
                    lg = self.forward(X_val)
                    te_loss = self._loss(lg, y_val).item()
                    accuracy_evaluator.data = {'true_y': y_val, 'pred_y': lg.argmax(1)}
                    te_acc = accuracy_evaluator.evaluate()
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(total / n)
            self.history['train_acc'].append(tr_acc)
            self.history['test_loss'].append(te_loss)
            self.history['test_acc'].append(te_acc)
            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % (total / n), 'Train Acc: %.4f' % tr_acc,
                      'Test Acc: %.4f' % te_acc)

    def test(self, X):
        nn.Module.train(self, False)
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.argmax(1)

    def run(self):
        print('method running...')
        print('--start training...')
        Xte = torch.FloatTensor(np.array(self.data['test']['X']))
        yte = torch.LongTensor(np.array(self.data['test']['y']))
        # the test set is only *monitored* each epoch (for the convergence plot);
        # it is never used to update weights.
        self.train(self.data['train']['X'], self.data['train']['y'], Xte, yte)
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
