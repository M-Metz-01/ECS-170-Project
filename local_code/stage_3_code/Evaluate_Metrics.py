
from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        if hasattr(y_true, 'numpy'):
            y_true = y_true.numpy()
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()
        out = {'accuracy': accuracy_score(y_true, y_pred)}
        for avg in ['weighted', 'macro', 'micro']:
            out['precision_' + avg] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            out['recall_' + avg] = recall_score(y_true, y_pred, average=avg, zero_division=0)
            out['f1_' + avg] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        return out
