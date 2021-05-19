from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
from os.path import join
from pathlib import Path

import pandas as pd
import numpy as np
import glob
import enum
import re
import os


class STLType(enum.Enum):
    ORIGINAL = 0
    TREND = 1
    SEASONAL = 2
    RESIDUAL = 3


class AdjacencyType(enum.Enum):
    CROSS = 0
    SIDE = 1
    ACROSS = 2
    ALL = 3


class CableSTL:
    def __init__(self, period, start, end, case_name):
        self.result_path = join(Path(__file__).parent.parent.absolute(), 'results')
        self.period = period
        self.start = start
        self.end = end
        self.case_name = case_name

    def get_day_stl(self, tension_df, str_date):
        cable_names = tension_df.columns.tolist()

        str_dict = dict()
        for cable in cable_names:
            str_df = self.get_cable_stl(tension_df, cable, str_date)
            str_dict.update({cable: str_df})

        return str_dict

    def get_cable_stl(self, tension_df, cable_name, str_date):
        start = datetime.strptime(str_date, '%Y-%m-%d') + timedelta(seconds=(self.start + 90) * 0.5)
        range = pd.date_range(start, periods=(self.end - self.start), freq='500ms')

        tension = tension_df.loc[:, cable_name].to_list()
        tension_series = pd.Series(tension[self.start:self.end], index=range, name=str)

        stl_df = self.get_stl(tension_series)
        self.save_stl(stl_df, cable_name, str_date)
        return stl_df

    def get_stl(self, tension):
        stl = STL(tension, period=self.period).fit()
        trend = stl.trend.to_numpy()
        seasonal = stl.seasonal.to_numpy()
        residual = stl.resid.to_numpy()
        stl_df = pd.DataFrame({'original': tension, 'trend': trend,
                               'seasonal': seasonal, 'residual': residual})
        return stl_df

    def save_stl(self, stl_df, cable_name, str_date):
        period_path = join(self.result_path, f'{self.case_name}_period_{self.period}')
        Path(period_path).mkdir(parents=True, exist_ok=True)

        result_path = join(period_path, f'{str_date}_{cable_name}.csv')
        stl_df.to_csv(result_path)
        print(f'saving stl data to {result_path}')


class CrossCorrelation:
    def __init__(self, result_name, date='', hourly=False):
        self.result_name = result_name
        self.result_path = join(Path(os.getcwd()), 'results', result_name)
        self.cc_path = join(Path(os.getcwd()), 'results', 'cross_correlation', result_name)
        self.result_path_list = glob.glob(join(self.result_path, f'{date}*.csv'))
        self.date_regex = r'(19|20)\d{2}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[0-1])'
        self.cable_regex = r'[A-Z]{3}[0-9]{2}'
        self.hourly = hourly

        self.stl_dict, self.date_list, self.cable_list = self.load_stl_results()

    def load_stl_results(self):
        stl_dict = dict()
        date_list = []
        cable_list = []

        for path in self.result_path_list:
            date_m = re.search(self.date_regex, path)
            cable_m = re.search(self.cable_regex, path)
            date = date_m.group(0)
            cable = cable_m.group(0)
            date_list.append(date)
            cable_list.append(cable)
            if self.hourly:
                for i in range(24):
                    if i == 0:
                        stl_df = pd.read_csv(path).iloc[:7200 - 90, 1:]
                    elif i < 23:
                        stl_df = pd.read_csv(path).iloc[(i*7200-90):((i*7200-90)+7200), 1:]
                    else:
                        stl_df = pd.read_csv(path).iloc[(i*7200-90):, 1:]
                    stl_df.reset_index(drop=True, inplace=True)
                    stl_dict.update({(date, cable, i): stl_df})
            else:
                stl_df = pd.read_csv(path).iloc[:, 1:]
                stl_dict.update({(date, cable): stl_df})

        date_list = sorted(list(set(date_list)))
        cable_list = sorted(list(set(cable_list)))
        return stl_dict, date_list, cable_list

    def load_day_dict(self, date):
        day_dict = dict()

        for cable_name in self.cable_list:
            stl_df = self.stl_dict[(date, cable_name)]
            day_dict.update({cable_name: stl_df})

        return day_dict

    def get_cross_correlation(self, date, cable_name_list=None, stl_type=STLType.RESIDUAL):
        day_dict = self.load_day_dict(date)

        if cable_name_list is None:
            cable_name_list = self.cable_list

        if len(cable_name_list) == 1:
            return []

        mean_list = []
        for i, cable1 in enumerate(cable_name_list):
            correlation_list = []

            for j, cable2 in enumerate(cable_name_list):
                if cable1 is cable2:
                    continue

                value1 = day_dict[cable1].iloc[:, stl_type.value].to_list()
                value2 = day_dict[cable2].iloc[:, stl_type.value].to_list()
                cross_correlation = np.correlate(value1, value2)
                correlation_list.append(cross_correlation)

            mean = sum(correlation_list) / float(len(correlation_list))
            mean_list.append(mean[0])

        return mean_list

    def get_temporal_cross_correlation(self, cable, stl_type):
        cc_df = pd.DataFrame(index=self.date_list, columns=[at.name for at in AdjacencyType])

        for date in self.date_list:
            mean_list = []
            for at in AdjacencyType:
                mean = self.get_adjacency_cross_correlation(date, cable, at, stl_type)
                if len(mean) == 0:
                    mean = np.nan
                else:
                    mean = mean[cable]
                mean_list.append(mean)
            cc_df.loc[date, :] = mean_list

        cc_path = join(self.cc_path, stl_type.name)
        Path(cc_path).mkdir(parents=True, exist_ok=True)
        cc_path = join(cc_path, f'{cable}.csv')
        cc_df.to_csv(cc_path)
        print(f'saving stl data to {cc_path}')
        return cc_df

    def get_adjacency_cross_correlation(self, date, cable_name, adjacency_type, stl_type):
        cable_list = [cable_name]

        if adjacency_type == AdjacencyType.CROSS:
            cable_list.extend(get_across_cable(cable_name))
            cable_list.extend(get_side_cables(cable_name))
        elif adjacency_type == AdjacencyType.SIDE:
            cable_list.extend(get_side_cables(cable_name))
        elif adjacency_type == AdjacencyType.ACROSS:
            cable_list.extend(get_across_cable(cable_name))
        else:
            cable_list = self.cable_list

        if 'SJX08' in cable_list:
            cable_list.remove('SJX08')
        elif 'SJS13' in cable_list:
            cable_list.remove('SJS13')
        elif 'SJX13' in cable_list:
            cable_list.remove('SJX13')

        return self.get_mean_cross_correlations(date, cable_list, stl_type)

    def get_mean_cross_correlations(self, date, cable_name_list=None, stl_type=STLType.RESIDUAL):
        if cable_name_list is None:
            cable_name_list = self.cable_list

        mean_list = self.get_cross_correlation(date, cable_name_list, stl_type)

        mean_dict = dict()
        for i, mean in enumerate(mean_list):
            mean_dict.update({cable_name_list[i]: mean})

        return mean_dict


def get_across_cable(cable_name):
    if cable_name[2] == 'S':
        return [f'SJX{cable_name[-2:]}']
    else:
        return [f'SJS{cable_name[-2:]}']


def get_side_cables(cable_name):
    number = int(cable_name[-2:])
    cable_names = []
    if number != 8:
        cable_names.append(f'{cable_name[:3]}{str(number - 1).zfill(2)}')
    elif number != 14:
        cable_names.append(f'{cable_name[:3]}{str(number + 1).zfill(2)}')

    return cable_names
