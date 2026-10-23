'''
Setting for data that is already split into a train file and a test file
(no train/test split and no cross validation needed).
'''

from local_code.base_class.setting import setting


class Setting_Train_Test_Pre_Partitioned(setting):
    test_dataset = None  # second dataset loader, for the test file

    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestDataset=None):
        super().prepare(sDataset, sMethod, sResult, sEvaluate)
        self.test_dataset = sTestDataset

    def load_run_save_evaluate(self):
        train = self.dataset.load()
        test = self.test_dataset.load()

        self.method.data = {'train': train, 'test': test}
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
