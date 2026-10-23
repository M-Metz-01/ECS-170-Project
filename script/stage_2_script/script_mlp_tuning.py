'''
Stage 2-4: improve the MLP (deeper network, other losses / optimizers / settings).
Run from inside script/stage_2_script/ :   python script_mlp_tuning.py

Model selection uses a validation set held out from the TRAINING data (last 10%), so the
testing set is only used once per configuration for reporting, never to choose the winner.
Writes: result/stage_2_result/tuning_results.csv, per-config convergence plots,
and best_model_convergence.png (best config retrained on the full training set).
'''
import sys, os, csv, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Curves import plot_convergence
import numpy as np
import torch

OUT = '../../result/stage_2_result/'
tr = Dataset_Loader('train', ''); tr.dataset_source_folder_path = '../../data/stage_2_data/'; tr.dataset_source_file_name = 'train.csv'
te = Dataset_Loader('test', '');  te.dataset_source_folder_path = '../../data/stage_2_data/';  te.dataset_source_file_name = 'test.csv'
train, test = tr.load(), te.load()

# fixed shuffled train/validation split of the training data
rng = np.random.RandomState(2)
perm = rng.permutation(len(train['y']))
n_val = len(perm) // 10
val_idx, fit_idx = perm[:n_val], perm[n_val:]
fit = {'X': train['X'][fit_idx], 'y': train['y'][fit_idx]}
val = {'X': train['X'][val_idx], 'y': train['y'][val_idx]}

base = dict(hidden_sizes=[128], activation='relu', loss_name='cross_entropy', optimizer_name='adam',
            learning_rate=1e-3, max_epoch=30, batch_size=128, dropout=0.0, batch_norm=False,
            weight_decay=0.0, lr_schedule=None)

# one-factor-at-a-time experiments around the baseline, then combined configs
experiments = [
    ('baseline_1x128',        {}),
    ('2_layers_256_128',      dict(hidden_sizes=[256, 128])),
    ('3_layers_512_256_128',  dict(hidden_sizes=[512, 256, 128])),
    ('4_layers_512_256_128_64', dict(hidden_sizes=[512, 256, 128, 64])),
    ('loss_label_smoothing',  dict(hidden_sizes=[256, 128], loss_name='label_smoothing')),
    ('loss_mse',              dict(hidden_sizes=[256, 128], loss_name='mse')),
    ('opt_sgd',               dict(hidden_sizes=[256, 128], optimizer_name='sgd', learning_rate=0.05)),
    ('opt_sgd_momentum',      dict(hidden_sizes=[256, 128], optimizer_name='sgd_momentum', learning_rate=0.02)),
    ('opt_rmsprop',           dict(hidden_sizes=[256, 128], optimizer_name='rmsprop', learning_rate=1e-3)),
    ('opt_adamw_wd',          dict(hidden_sizes=[256, 128], optimizer_name='adamw', weight_decay=1e-2)),
    ('act_tanh',              dict(hidden_sizes=[256, 128], activation='tanh')),
    ('act_gelu',              dict(hidden_sizes=[256, 128], activation='gelu')),
    ('dropout_0.2',           dict(hidden_sizes=[512, 256, 128], dropout=0.2)),
    ('batchnorm',             dict(hidden_sizes=[512, 256, 128], batch_norm=True)),
    ('combo_A_3L_bn_drop_adamw_cos',
        dict(hidden_sizes=[512, 256, 128], batch_norm=True, dropout=0.2, optimizer_name='adamw',
             weight_decay=1e-2, lr_schedule='cosine', max_epoch=40)),
    ('combo_B_3L_drop_adam_cos_ls',
        dict(hidden_sizes=[512, 256, 128], dropout=0.2, lr_schedule='cosine', loss_name='label_smoothing', max_epoch=40)),
]

ev = Evaluate_Metrics('metrics', '')
rows, best = [], None
for name, over in experiments:
    cfg = dict(base); cfg.update(over)
    print('\n===== %s =====' % name)
    m = Method_MLP(name, '', **cfg)
    m.data = {'train': fit, 'test': val}          # 'test' here = validation split
    res = m.run()
    ev.data = res; vs = ev.evaluate()
    # now report on the real test set with the same trained model
    ev.data = {'true_y': test['y'], 'pred_y': m.test(test['X'])}; ts = ev.evaluate()
    row = {'name': name, 'val_acc': vs['accuracy'], 'test_acc': ts['accuracy'],
           'test_f1_weighted': ts['f1_weighted'], 'test_f1_macro': ts['f1_macro'], 'test_f1_micro': ts['f1_micro'],
           'test_prec_weighted': ts['precision_weighted'], 'test_rec_weighted': ts['recall_weighted'], 'config': str(over)}
    rows.append(row)
    print('val acc %.4f | test acc %.4f | test weighted F1 %.4f' % (vs['accuracy'], ts['accuracy'], ts['f1_weighted']))
    if best is None or vs['accuracy'] > best[0]:
        best = (vs['accuracy'], name, cfg)

with open(OUT + 'tuning_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# retrain the config with the best VALIDATION accuracy on the full training set; report on test
print('\n===== best by validation: %s (val acc %.4f) =====' % (best[1], best[0]))
torch.manual_seed(2)
final = Method_MLP('best', '', **best[2])
final.data = {'train': train, 'test': test}
res = final.run()
ev.data = res; s = ev.evaluate()
print('************ FINAL TEST PERFORMANCE (best config) ************')
for k, v in s.items():
    print('%-20s %.4f' % (k, v))
plot_convergence(final.history, OUT + 'best_model_convergence.png', 'Best MLP: ' + best[1])
with open(OUT + 'best_model_summary.txt', 'w') as f:
    f.write('best config: %s\n%s\n\n' % (best[1], best[2]))
    for k, v in s.items():
        f.write('%-20s %.4f\n' % (k, v))
print('saved tuning_results.csv, best_model_convergence.png, best_model_summary.txt')
