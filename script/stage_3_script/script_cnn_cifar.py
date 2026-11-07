
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Pre_Partitioned import Setting_Train_Test_Pre_Partitioned
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Plot_Curves import plot_convergence
import numpy as np
import torch

#---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
#------------------------------------------------------

# ---- objection initialization section ---------------
#one loader: the CIFAR file holds both the train and the test images
data_obj = Dataset_Loader('cifar', '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = 'CIFAR'

# baseline: 3 conv blocks (32, 64, 128 channels) with 2 convs each + batch norm, 3x3 kernels, max pooling,
# 1 hidden layer (256), dropout 0.3, adam, lr 1e-3 with cosine schedule
method_obj = Method_CNN_CIFAR('CNN CIFAR', '')

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
result_obj.result_destination_file_name = 'prediction_result'
result_obj.fold_count = 'baseline'

setting_obj = Setting_Train_Test_Pre_Partitioned('pre-partitioned train/test', '')
evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
# ------------------------------------------------------

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
scores = setting_obj.load_run_save_evaluate()
print('************ Overall Performance (testing set) ************')
for k, v in scores.items():
    print('%-20s %.4f' % (k, v))
plot_convergence(method_obj.history, '../../result/stage_3_result/cifar_convergence.png',
                 'CNN on CIFAR-10 (3 conv blocks x 2 convs: 32, 64, 128)')
print('learning curves saved to result/stage_3_result/cifar_convergence.png')
print('************ Finish ************')
# ------------------------------------------------------
