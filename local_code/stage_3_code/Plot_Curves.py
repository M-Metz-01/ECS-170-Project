

#imports
import matplotlib
matplotlib.use('Agg')   #save plots to files without opening a window
import matplotlib.pyplot as plt


def plot_convergence(history, save_path, title=''):
    #draws training/testing loss (left) and training/testing accuracy (right) and saves the figure
    ep = history['epoch']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    #loss curves
    ax[0].plot(ep, history['train_loss'], label='training loss')
    ax[0].plot(ep, history['test_loss'], label='testing loss')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].set_title('Loss convergence')
    ax[0].legend(); ax[0].grid(alpha=0.3)
    #accuracy curves
    ax[1].plot(ep, history['train_acc'], label='training accuracy')
    ax[1].plot(ep, history['test_acc'], label='testing accuracy')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('accuracy'); ax[1].set_title('Accuracy convergence')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_compare(histories, save_path, title=''):
    #draws the validation accuracy curve of every configuration on one plot so they can be compared
    #histories: {config name: history dict}
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, h in histories.items():
        ax.plot(h['epoch'], h['test_acc'], label=name)
    ax.set_xlabel('epoch'); ax.set_ylabel('validation accuracy'); ax.set_title(title)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
