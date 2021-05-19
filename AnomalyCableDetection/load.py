from AnomalyCableDetection.preprocess import Preprocessor
from pathlib import Path
from os.path import join

import pandas as pd
import glob
import re


class Loader:
    def __init__(self, iqr, pre_version):
        self.iqr = iqr
        self.pre_version = pre_version

        self.data_path = join(Path(__file__).parent.parent.absolute(), 'data')
        self.date_regex = r'(19|20)\d{2}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[0-1])'
        self.preprocessor = Preprocessor(iqr, pre_version)
        self.df_dict = self.load_preprocessed_dataset()

    def load_raw_dataset(self):
        df_dict = self.preprocessor.load_raw_dataset()
        return df_dict

    def load_preprocessed_dataset(self):
        if self.iqr:
            data_path = join(self.data_path, 'csv', 'pre_IQR')
        else:
            if self.pre_version == 0:
                pre_name = ''
            elif self.pre_version == 1:
                pre_name = '_1'
            else:
                pre_name = ''
            data_path = join(self.data_path, 'csv', f'preprocessed{pre_name}')

        file_list = glob.glob(join(data_path, f'*.csv'))

        if len(file_list) == 0:
            df_dict = self.preprocessor.preprocess()
        else:
            df_dict = dict()
            for file_path in file_list:
                df = pd.read_csv(file_path, index_col=None)
                match = re.search(self.date_regex, file_path)
                df_dict.update({match.group(0): df})

        self.df_dict = df_dict
        return df_dict

    def get_dates(self):
        dates = [date for date in self.df_dict]
        return dates

    def get_cables(self):
        dates = self.get_dates()
        df = self.df_dict[dates[0]]
        return df.columns.tolist()

    def load_day_tension(self, date):
        return self.df_dict[date]


if __name__ == '__main__':
    loader = Loader(iqr=False, pre_version=1)
    tension_dict = loader.load_preprocessed_dataset()
