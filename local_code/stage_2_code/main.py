import os
import matplotlib.pyplot as plt
from .Dataset_Loader import Dataset_Loader
from .Method_MLP import Method_MLP
from .Result_Saver import Result_Saver
from .Setting_Train_Test_Pre_Partitioned import Setting_Train_Test_Pre_Partitioned
from .Evaluate_Metrics import Evaluate_Metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'stage_2_data') + '/'
RESULT_DIR = os.path.join(HERE, '..', '..', 'result', 'stage_2_result')
os.makedirs(RESULT_DIR, exist_ok=True)

train_loader = Dataset_Loader('train', '')
train_loader.dataset_source_folder_path = DATA_DIR
train_loader.dataset_source_file_name = 'train.csv'

test_loader = Dataset_Loader('test', '')
test_loader.dataset_source_folder_path = DATA_DIR
test_loader.dataset_source_file_name = 'test.csv'

model = Method_MLP('mlp', '', hidden_sizes=[256, 128], max_epoch=30)

saver = Result_Saver('saver', '')
saver.result_destination_folder_path = RESULT_DIR + '/MLP_'
saver.result_destination_file_name = 'prediction_result'
saver.fold_count = 1

evaluator = Evaluate_Metrics('metrics', '')

settings = Setting_Train_Test_Pre_Partitioned('pre-partitioned', '')
settings.prepare(train_loader, model, saver, evaluator, test_loader)
print('hidden:', model.hidden_sizes, 'epochs:', model.max_epoch, 'optimizer:', model.optimizer_name)

scores = settings.load_run_save_evaluate()

for name, value in scores.items():
    print(f'{name}: {value:.4f}')

print('pred:', saver.data['pred_y'][:10])
print('true:', saver.data['true_y'][:10])

history = model.history

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(history['epoch'], history['train_loss'], label='train')
ax1.plot(history['epoch'], history['test_loss'], label='test')
ax1.set_xlabel('epoch'); ax1.set_ylabel('loss'); ax1.legend()
ax2.plot(history['epoch'], history['train_acc'], label='train')
ax2.plot(history['epoch'], history['test_acc'], label='test')
ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy'); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'convergence.png'), dpi=150)
