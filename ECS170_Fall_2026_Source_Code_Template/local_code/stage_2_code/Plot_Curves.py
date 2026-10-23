'''
Helper that draws the learning convergence curves (loss and accuracy vs. epoch).
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_convergence(history, save_path, title=''):
    ep = history['epoch']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history['train_loss'], label='training loss')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].set_title('Loss convergence')
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, history['train_acc'], label='training accuracy')
    ax[1].plot(ep, history['test_acc'], label='testing accuracy')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('accuracy'); ax[1].set_title('Accuracy convergence')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
