import configparser
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, config: configparser.ConfigParser) -> None:
        self.config = config

    def read_data_from_config(self) -> pd.DataFrame:
        data = []
        sep = self.config['data.raw']['sep']
        for k, path in self.config['data.raw'].items():
            if 'data' in k:
                data.append(pd.read_csv(path, sep=sep))
        data = pd.concat(data)
        return data

    def split(self) -> list[pd.DataFrame]:
        df_data = self.read_data_from_config()
        return train_test_split(
            df_data.iloc[:, :-1], df_data.iloc[:, -1],
            train_size=float(self.config['data.raw']['train_pers']), random_state=42
        )

    def split_and_save(self) -> None:
        X_train, X_test, y_train, y_test = self.split()
        X_train.to_csv(self.config['data.prep']['train_x'], index=False)
        y_train.to_csv(self.config['data.prep']['train_y'], index=False)
        X_test.to_csv(self.config['data.prep']['test_x'], index=False)
        y_test.to_csv(self.config['data.prep']['test_y'], index=False)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    print("Splitting data")
    DataSplitter(config).split_and_save()
    print("Data is ready")
