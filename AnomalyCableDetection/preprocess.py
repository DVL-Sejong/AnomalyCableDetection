from scipy.io import loadmat
from pathlib import Path
from os.path import join

from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy import signal

import pandas as pd
import numpy as np
import glob
import os
import re


class MatConverter:
    def __init__(self):
        self.data_path = join(Path(__file__).parent.parent.absolute(), 'data')
        self.date_regex = r'(19|20)\d{2}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[0-1])'

    def mat_to_csv(self):
        mat_dict = self.get_mat_files()

        df_dict = dict()
        for str_date in mat_dict:
            cable_names = [mat_dict[str_date][1][0][i][0] for i in range(14)]
            day_tension = self.get_day_tension(mat_dict[str_date][0])
            tension_df = pd.DataFrame(day_tension, columns=cable_names)
            csv_path = join(self.data_path, 'csv', 'raw', f'{str_date}.csv')
            tension_df.to_csv(csv_path, index=False)
            df_dict.update({str_date: tension_df})
            print(f'converting mat file into csv: {csv_path}')

        return df_dict

    def get_file_names(self, extension):
        data_path = os.path.join(self.data_path, extension)
        file_names = sorted(glob.glob(join(data_path, f'*.{extension}')))
        return file_names

    def get_mat_files(self):
        mat_file_names = self.get_file_names('mat')

        mat_dict = dict()
        for file_name in mat_file_names:
            mat_file = loadmat(file_name)['CF'][0][0]
            match = re.search(self.date_regex, file_name)
            mat_name = match.group(0)
            mat_dict.update({mat_name: mat_file})

        return mat_dict

    def get_day_tension(self, mat_file):
        day_tension = []
        for cable_index in range(14):
            cable_tension = [elem[cable_index] for elem in mat_file]
            day_tension.append(cable_tension)

        return np.array(day_tension).T


class Preprocessor:
    def __init__(self, iqr, pre_version):
        self.iqr = iqr
        self.pre_version = pre_version

        self.data_path = join(Path(__file__).parent.parent.absolute(), 'data')
        self.date_regex = r'(19|20)\d{2}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[0-1])'

    def preprocess(self, cable_name='SJS13', window=3600):
        raw_dict = self.load_raw_dataset()

        preprocessed_dict = dict()
        for str_date in raw_dict:
            preprocessed_df = self.remove_anomaly(raw_dict[str_date], cable_name)
            preprocessed_df = self.apply_moving_average(preprocessed_df, window, subtract=True)
            data_path = join(self.data_path, 'csv', 'preprocessed', f'{str_date}.csv')
            preprocessed_df.to_csv(data_path, index=False)
            preprocessed_dict.update({str_date: preprocessed_df})
            print(f'saving preprocessed dataset to {data_path}')

        return preprocessed_dict

    def load_raw_dataset(self):
        if self.iqr:
            data_path = join(self.data_path, 'csv', 'pre_IQR')
        else:
            data_path = join(self.data_path, 'csv', 'raw')
        file_list = glob.glob(join(data_path, f'*.csv'))

        if len(file_list) == 0:
            converter = MatConverter()
            df_dict = converter.mat_to_csv()
        else:
            df_dict = self.load_csv_files(file_list)

        return df_dict

    def load_csv_files(self, path_list):
        df_dict = dict()

        for csv_path in path_list:
            df = pd.read_csv(csv_path, index_col=None)
            match = re.search(self.date_regex, csv_path)
            name = match.group(0)
            df_dict.update({name: df})

        return df_dict

    def remove_anomaly(self, tension_df, cable_name):
        new_tension_df = tension_df.copy(deep=True)
        q25, q75 = np.quantile(new_tension_df[cable_name], [0.25, 0.75])
        upper_threshold = q75 - 1.5 * (q75 - q25)
        lower_threshold = q25 + 1.5 * (q75 - q25)

        new_tension_df[cable_name][new_tension_df[cable_name] < upper_threshold] = np.nan
        new_tension_df[cable_name][new_tension_df[cable_name] > lower_threshold] = np.nan

        new_tension_df = new_tension_df.interpolate(method='linear')
        new_tension_df = new_tension_df.iloc[90:-90, :]
        new_tension_df.reset_index(drop=True, inplace=True)
        return new_tension_df

    def low_pass_filter(self, tension_df, order, cutoff, fs, low):
        cable_names = tension_df.columns.tolist()
        index = tension_df.index.to_list()[:-order]
        new_tension_df = tension_df.copy(deep=True)
        filtered_df = pd.DataFrame(columns=cable_names, index=index)

        for i, cable in enumerate(cable_names):
            b = signal.firwin(order, cutoff=cutoff, fs=fs, pass_zero='lowpass')
            x = signal.lfilter(b, [1.0], new_tension_df.loc[:, cable].to_list())
            if low:
                values = new_tension_df.loc[order:, cable].to_list() - x[order:]
            else:
                values = x[order:]
            filtered_df.loc[:, cable] = values

        return filtered_df

    def apply_moving_average(self, tension_df, window, subtract):
        new_tension_df = tension_df.copy(deep=True)
        cable_names = new_tension_df.columns.tolist()

        for i, cable in enumerate(cable_names):
            cable_values = new_tension_df.loc[:, cable].to_list()
            smoothed = self.smooth(cable_values, window)
            if subtract:
                new_tension_df.loc[:, cable] -= smoothed
            else:
                new_tension_df.loc[:, cable] = smoothed

        return new_tension_df

    def cy_final_preprocess(self, cable_data):
        """
        preprocess dataset, made by ycy
        preprocessed dataset through this method is saved in 'pre1' directory.
        :param cable_data: shape in (10, 14, 172620), (10 days, 14 cables, 172620 tension values)
        :return: preprocessed rolled cable dataset
        """
        cable_rolling = []
        cable_diff_rolling = []
        for i in range(cable_data.shape[0]):
            rolling_datas = []
            diff_rolling_datas = []
            for j in range(cable_data.shape[1]):
                rollings = []
                data_rollings = []
                for k in range(24):
                    data = cable_data[i][j][k * 7200:(k + 1) * 7200]
                    rolling, data_rolling = cy_preprocess(data)
                    rollings.append(rolling)
                    data_rollings.append(data_rolling)
                rolling = np.concatenate(rollings)
                data_rolling = np.concatenate(data_rollings)
                rolling_datas.append(rolling)
                diff_rolling_datas.append(data_rolling)
            rolling_datas = np.array(rolling_datas)
            diff_rolling_datas = np.array(diff_rolling_datas)
            cable_rolling.append(rolling_datas)
            cable_diff_rolling.append(diff_rolling_datas)
            print(rolling_datas.shape, diff_rolling_datas.shape)
        cable_rolling = np.array(cable_rolling)
        cable_diff_rolling = np.array(cable_diff_rolling)
        return cable_rolling

    def smooth(self, array, size):
        out = np.convolve(array, np.ones(size, dtype=int), 'valid') / size
        r = np.arange(1, size - 1, 2)
        start = np.cumsum(array[:size - 1][::2]) / r
        stop = (np.cumsum(array[:-size:-1])[::2] / r)[::-1]
        return np.concatenate((start, out, stop))


def cy_preprocess(data):
    peaks, _ = find_peaks(data, height=(np.mean(data) - np.std(data), np.mean(data) + np.std(data)))
    peak_data = make_nan_array(len(data), peaks, data[peaks])
    peak = data[peaks]
    median_peak = median_filter(peak, 50)
    median_peak_data = make_nan_array(len(data), peaks, median_peak)

    inter = pd.Series(median_peak_data).interpolate()
    rolling = pd.Series(inter).rolling(1200, min_periods=1).mean()

    return rolling, data - rolling


def make_nan_array(length, index_list, values):
    data = np.empty(length)
    data[:] = np.nan
    np.put(data, index_list, values)
    idx = np.argwhere(~np.isnan(data))[0][0]
    data[0] = data[idx]
    return data


if __name__ == '__main__':
    converter = MatConverter()
    df_dict = converter.mat_to_csv()
