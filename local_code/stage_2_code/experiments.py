import os
import pandas as pd
import matplotlib.pyplot as plt

from .Dataset_Loader import Dataset_Loader
from .Method_MLP import Method_MLP
from .Evaluate_Metrics import Evaluate_Metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'stage_2_data') + '/'
OUT_DIR = os.path.join(HERE, '..', '..', 'result', 'stage_2_result', 'experiments')
os.makedirs(OUT_DIR, exist_ok=True)

configs = {
    'baseline':     dict(hidden_sizes=[256, 128]),
    'deeper':       dict(hidden_sizes=[512, 256, 128]),
    'dropout_bn':   dict(hidden_sizes=[512, 256, 128], dropout=0.2, batch_norm=True),
    'sgd_momentum': dict(hidden_sizes=[256, 128], optimizer_name='sgd_momentum', learning_rate=0.05),
    'label_smooth': dict(hidden_sizes=[256, 128], loss_name='label_smoothing'),
    'mse_loss':     dict(hidden_sizes=[256, 128], loss_name='mse'),
    'adamw_cosine': dict(hidden_sizes=[512, 256], optimizer_name='adamw', weight_decay=1e-4, lr_schedule='cosine'),
}

train_loader = Dataset_Loader('train', '')
train_loader.dataset_source_folder_path = DATA_DIR
train_loader.dataset_source_file_name = 'train.csv'

test_loader = Dataset_Loader('test', '')
test_loader.dataset_source_folder_path = DATA_DIR
test_loader.dataset_source_file_name = 'test.csv'

data = {'train': train_loader.load(), 'test': test_loader.load()}

evaluator = Evaluate_Metrics('metrics', '')
rows, histories = [], {}

for name, cfg in configs.items():
    print(f'\n=== {name} ===')

    model = Method_MLP(name, '', max_epoch=30, **cfg)
    model.data = data

    result = model.run()

    evaluator.data = result
    scores = evaluator.evaluate()
    rows.append({'config': name, **scores})
    histories[name] = result['history']

    h = result['history']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(h['epoch'], h['train_loss'], label='train')
    ax1.plot(h['epoch'], h['test_loss'], label='test')
    ax1.set_title(f'{name} loss'); ax1.set_xlabel('epoch'); ax1.legend()
    ax2.plot(h['epoch'], h['train_acc'], label='train')
    ax2.plot(h['epoch'], h['test_acc'], label='test')
    ax2.set_title(f'{name} accuracy'); ax2.set_xlabel('epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=150)
    plt.close()

table = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
cols = ['config', 'accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']
print('\n', table[cols].to_string(index=False, float_format='%.4f'))
table.to_csv(os.path.join(OUT_DIR, 'comparison.csv'), index=False)

plt.figure(figsize=(8, 5))
for name, h in histories.items():
    plt.plot(h['epoch'], h['test_acc'], label=name)
plt.xlabel('epoch'); plt.ylabel('test accuracy'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'all_test_acc.png'), dpi=150)
