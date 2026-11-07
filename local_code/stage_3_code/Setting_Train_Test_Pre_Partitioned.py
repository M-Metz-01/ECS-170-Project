

#imports
from local_code.base_class.setting import setting


class Setting_Train_Test_Pre_Partitioned(setting):

    def load_run_save_evaluate(self):
        #load both splits from the one dataset file
        loaded_data = self.dataset.load()

        #train on the train split and predict the test split
        self.method.data = {'train': loaded_data['train'], 'test': loaded_data['test']}
        learned_result = self.method.run()

        #save the predictions
        self.result.data = learned_result
        self.result.save()

        #score the predictions
        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
