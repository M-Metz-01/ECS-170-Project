'''
Stage 2-3: baseline MLP on the pre-partitioned train/test data.
Run from inside script/stage_2_script/ :   python script_mlp.py
(or from the project root with PYTHONPATH=. )
'''
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Pre_Partitioned import Setting_Train_Test_Pre_Partitioned
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Curves import plot_convergence
import numpy as np
import torch

#---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
#------------------------------------------------------

# ---- objection initialization section ---------------
train_obj = Dataset_Loader('mnist-train', '')
train_obj.dataset_source_folder_path = '../../data/stage_2_data/'
train_obj.dataset_source_file_name = 'train.csv'

test_obj = Dataset_Loader('mnist-test', '')
test_obj.dataset_source_folder_path = '../../data/stage_2_data/'
test_obj.dataset_source_file_name = 'test.csv'

# baseline: one hidden layer (128 units), ReLU, cross-entropy, Adam, lr 1e-3
method_obj = Method_MLP('multi-layer perceptron', '', hidden_sizes=[128], activation='relu',
                        loss_name='cross_entropy', optimizer_name='adam', learning_rate=1e-3,
                        max_epoch=30, batch_size=128)

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
result_obj.result_destination_file_name = 'prediction_result'
result_obj.fold_count = 'baseline'

setting_obj = Setting_Train_Test_Pre_Partitioned('pre-partitioned train/test', '')
evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
# ------------------------------------------------------

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj, test_obj)
setting_obj.print_setup_summary()
scores = setting_obj.load_run_save_evaluate()
print('************ Overall Performance (testing set) ************')
for k, v in scores.items():
    print('%-20s %.4f' % (k, v))
plot_convergence(method_obj.history, '../../result/stage_2_result/baseline_convergence.png',
                 'Baseline MLP (1 hidden layer, 128 units)')
print('convergence plot saved to result/stage_2_result/baseline_convergence.png')
print('************ Finish ************')
# ------------------------------------------------------
