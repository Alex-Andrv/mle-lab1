import configparser
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score


class Predictor:
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, clf) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.clf = clf

    @classmethod
    def from_config(cls, config: configparser.ConfigParser):
        X_test = pd.read_csv(config['data.prep']['test_x'])
        y_test = pd.read_csv(config['data.prep']['test_y'])
        clf = load(config['train']['save_path'])
        return cls(X_test, y_test, clf)

    def calc_accuracy(self) -> float:
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    predictor = Predictor.from_config(config)
    print(predictor.calc_accuracy())
