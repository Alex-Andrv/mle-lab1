import configparser
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


class Trainer:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, clf_cfg: dict) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.clf_cfg = clf_cfg

    @classmethod
    def from_config(cls, config: configparser.ConfigParser):
        X_train = pd.read_csv(config['data.prep']['train_x'])
        y_train = pd.read_csv(config['data.prep']['train_y'])
        clf_cfg = config['train']
        return cls(X_train, y_train, dict(clf_cfg))

    def train(self) -> None:
        if self.clf_cfg['model_name'] == 'random_forest':
            clf = RandomForestClassifier(n_estimators=int(self.clf_cfg['n_estimators']),
                                         random_state=int(self.clf_cfg['random_state']), )
        else:
            clf = RandomForestClassifier()
        print('Training...')
        clf = clf.fit(self.X_train, self.y_train)
        print('Saving model...')
        dump(clf, self.clf_cfg['save_path'])
        print('Training finished')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    trainer = Trainer.from_config(config)
    trainer.train()
