import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Plot_Curves import plot_convergence, plot_compare
import numpy as np
import torch

#---- parameter section -------------------------------
#which dataset to tune (first command line argument, default mnist)
ds = sys.argv[1].lower() if len(sys.argv) > 1 else 'mnist'
OUT = '../../result/stage_3_result/'

#per dataset: file name, CNN class, and the num of epochs used for each tuning run
#(fewer epochs than the full baseline runs so that trying many configs doesn't take hours on a cpu)
DATASETS = {
    'mnist': ('MNIST', Method_CNN_MNIST, 5),
    'orl':   ('ORL',   Method_CNN_ORL,   30),
    'cifar': ('CIFAR', Method_CNN_CIFAR, 5),
}

#one-factor-at-a-time experiments around each dataset's baseline (the defaults in its Method_CNN_<dataset> class)
#each entry: (name, settings that are different from the baseline)
EXPERIMENTS = {
    'mnist': [
        ('baseline_2blocks_32_64',  {}),
        ('depth_1block_32',         dict(conv_channels=[32])),
        ('depth_3blocks_32_64_128', dict(conv_channels=[32, 64, 128])),
        ('kernel_5x5',              dict(kernel_size=5)),
        ('kernel_7x7',              dict(kernel_size=7)),
        ('padding_0',               dict(padding=0)),
        ('stride_2_no_pool',        dict(stride=2, pooling=None)),
        ('pool_avg',                dict(pooling='avg')),
        ('hidden_32',               dict(hidden_sizes=[32])),
        ('hidden_512',              dict(hidden_sizes=[512])),
        ('loss_label_smoothing',    dict(loss_name='label_smoothing')),
        ('loss_mse',                dict(loss_name='mse')),
        ('opt_sgd_momentum',        dict(optimizer_name='sgd_momentum', learning_rate=0.02)),
        ('batchnorm',               dict(batch_norm=True)),
        ('act_tanh',                dict(activation='tanh')),
    ],
    'orl': [
        ('baseline_3blocks_16_32_64', {}),
        ('depth_2blocks_16_32',     dict(conv_channels=[16, 32])),
        ('depth_4blocks_16_32_64_128', dict(conv_channels=[16, 32, 64, 128])),
        ('kernel_5x5',              dict(kernel_size=5)),
        ('kernel_7x7',              dict(kernel_size=7)),
        ('padding_0',               dict(padding=0)),
        ('stride_2_no_pool',        dict(stride=2, pooling=None)),
        ('pool_avg',                dict(pooling='avg')),
        ('hidden_64',               dict(hidden_sizes=[64])),
        ('hidden_512',              dict(hidden_sizes=[512])),
        ('no_dropout',              dict(dropout=0.0)),
        ('loss_label_smoothing',    dict(loss_name='label_smoothing')),
        ('loss_mse',                dict(loss_name='mse')),
        ('opt_sgd_momentum',        dict(optimizer_name='sgd_momentum', learning_rate=0.01)),
        ('batchnorm',               dict(batch_norm=True)),
    ],
    'cifar': [
        ('baseline_3blocks_x2_bn',  {}),
        ('depth_2blocks_32_64',     dict(conv_channels=[32, 64])),
        ('1_conv_per_block',        dict(convs_per_block=1)),
        ('kernel_5x5',              dict(kernel_size=5)),
        ('stride_2_no_pool',        dict(stride=2, pooling=None, convs_per_block=1)),
        ('pool_avg',                dict(pooling='avg')),
        ('hidden_64',               dict(hidden_sizes=[64])),
        ('hidden_512',              dict(hidden_sizes=[512])),
        ('no_batchnorm',            dict(batch_norm=False)),
        ('loss_label_smoothing',    dict(loss_name='label_smoothing')),
        ('opt_sgd_momentum',        dict(optimizer_name='sgd_momentum', learning_rate=0.05)),
        ('augmentation',            dict(augment=True)),
    ],
}
#------------------------------------------------------

#load the dataset (both splits come from the same file)
file_name, CNN, tune_epochs = DATASETS[ds]
data_obj = Dataset_Loader(ds, '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = file_name
loaded = data_obj.load()
train, test = loaded['train'], loaded['test']

# fixed train/validation split of the training data: hold out 10% of every class
# (stratified so ORL, with only 9 images per person, still keeps 1 image of each person for validation)
rng = np.random.RandomState(2)
val_idx = []
for c in np.unique(train['y']):
    idx = rng.permutation(np.where(train['y'] == c)[0])
    val_idx.extend(idx[:max(1, len(idx) // 10)])
val_idx = np.array(val_idx)
fit_idx = np.setdiff1d(np.arange(len(train['y'])), val_idx)
fit = {'X': train['X'][fit_idx], 'y': train['y'][fit_idx]}
val = {'X': train['X'][val_idx], 'y': train['y'][val_idx]}
print('fit images:', len(fit_idx), 'validation images:', len(val_idx))

ev = Evaluate_Metrics('metrics', '')
rows, histories, best = [], {}, None
#train and score each configuration
for name, over in EXPERIMENTS[ds]:
    cfg = dict(max_epoch=tune_epochs); cfg.update(over)
    print('\n===== %s =====' % name)
    m = CNN(name, '', **cfg)
    m.data = {'train': fit, 'test': val}          # 'test' here = validation split
    res = m.run()
    ev.data = res; vs = ev.evaluate()
    # now report on the real test set with the same trained model
    ev.data = {'true_y': test['y'], 'pred_y': m.test(test['X'])}; ts = ev.evaluate()
    n_params = sum(p.numel() for p in m.parameters())
    row = {'name': name, 'val_acc': vs['accuracy'], 'test_acc': ts['accuracy'],
           'test_f1_weighted': ts['f1_weighted'], 'test_f1_macro': ts['f1_macro'], 'test_f1_micro': ts['f1_micro'],
           'test_prec_weighted': ts['precision_weighted'], 'test_rec_weighted': ts['recall_weighted'],
           'num_params': n_params, 'config': str(over)}
    rows.append(row)
    histories[name] = m.history
    print('val acc %.4f | test acc %.4f | test weighted F1 %.4f | params %d'
          % (vs['accuracy'], ts['accuracy'], ts['f1_weighted'], n_params))
    #keep the config with the best validation accuracy
    if best is None or vs['accuracy'] > best[0]:
        best = (vs['accuracy'], name, cfg)

#save the table of results and one plot comparing every config's validation curve
with open(OUT + 'tuning_results_%s.csv' % ds, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
plot_compare(histories, OUT + 'tuning_compare_%s.png' % ds, 'CNN configurations on %s (validation accuracy)' % file_name)

# retrain the config with the best VALIDATION accuracy on the full training set; report on test
print('\n===== best by validation: %s (val acc %.4f) =====' % (best[1], best[0]))
torch.manual_seed(2)
final = CNN('best', '', **best[2])
final.data = {'train': train, 'test': test}
res = final.run()
ev.data = res; s = ev.evaluate()
print('************ FINAL TEST PERFORMANCE (best config) ************')
for k, v in s.items():
    print('%-20s %.4f' % (k, v))
plot_convergence(final.history, OUT + 'best_model_convergence_%s.png' % ds, 'Best CNN on %s: %s' % (file_name, best[1]))
with open(OUT + 'best_model_summary_%s.txt' % ds, 'w') as f:
    f.write('best config: %s\n%s\n\n' % (best[1], best[2]))
    for k, v in s.items():
        f.write('%-20s %.4f\n' % (k, v))
print('saved tuning_results_%s.csv, tuning_compare_%s.png, best_model_convergence_%s.png, best_model_summary_%s.txt'
      % (ds, ds, ds, ds))
