from matplotlib import pyplot as plt
from sklearn import mixture
from os.path import join
from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import glob
import re


class CableGMM:
    def __init__(self, sigma_period=60, removing=True, hourly=False, hour=None):
        self.sigma_period = sigma_period
        self.removing = removing
        self.hourly = hourly
        self.hour = hour

        self.data_path = join(Path(__file__).parent.parent.absolute(), 'data', 'csv', 'preprocessed_1')
        self.path_list = glob.glob(join(self.data_path, '*.csv'))
        self.date_regex = r'(19|20)\d{2}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[0-1])'
        self.cable_dict, self.date_list = self.init_cable_dict()
        self.scatter_dict = self.init_scatter_dict()

    def init_cable_dict(self):
        date_list = []
        cable_dict = dict()

        for path in self.path_list:
            m = re.search(self.date_regex, path)
            date = m.group(0)
            date_list.append(date)
            cable_df = pd.read_csv(path, index_col=0)
            if self.hourly:
                cable_df = self.get_hour_df(cable_df)

            cable_list = cable_df.columns.tolist()
            if self.removing:
                cable_list.remove('SJX08')
                cable_list.remove('SJS13')
                cable_list.remove('SJX13')
            cable_df = cable_df[cable_list]
            cable_dict.update({date: cable_df})

        return cable_dict, sorted(date_list)

    def get_hour_df(self, cable_df):
        if self.hour == 0:
            cable_df = cable_df.iloc[:7200 - 90, 1:]
        elif self.hour < 23:
            cable_df = cable_df.iloc[(self.hour * 7200 - 90):((self.hour * 7200 - 90) + 7200), 1:]
        else:
            cable_df = cable_df.iloc[(self.hour * 7200 - 90):, 1:]
        cable_df.reset_index(drop=True, inplace=True)
        return cable_df

    def init_scatter_dict(self):
        tmp_df = self.cable_dict[self.date_list[0]]
        length = len(tmp_df.index.to_list())

        scatter_dict = dict()
        for date in self.date_list:
            scatter_df = self.get_scatter_dict(date, end=length)
            scatter_dict.update({date: scatter_df})

        return scatter_dict

    def get_scatter_dict(self, date, start=0, end=172620):
        tension_df = self.cable_dict[date]
        cable_numbers = get_dup_cable_numbers(tension_df)

        single_len = end - start
        index_len = single_len * len(cable_numbers)

        scatter_df = pd.DataFrame(index=[i for i in range(index_len)], columns=['x', 'y', 'cable'])
        for i, cable in enumerate(cable_numbers):
            cable_number = str(cable).zfill(2)
            x = tension_df.loc[start:end - 1, f'SJS{cable_number}'].to_list()
            y = tension_df.loc[start:end - 1, f'SJX{cable_number}'].to_list()
            scatter_df.loc[i * single_len:(i + 1) * single_len - 1, 'x'] = x
            scatter_df.loc[i * single_len:(i + 1) * single_len - 1, 'y'] = y
            scatter_df.loc[i * single_len:(i + 1) * single_len - 1, 'cable'] = f'SJ{cable_number}'

        return scatter_df

    def get_sigma_list(self, date, cable_number):
        day_df = self.cable_dict[date]
        length = len(day_df.index.to_list())

        sigma_list = []
        for i in range(0, length, self.sigma_period):
            upside = day_df.loc[i:i + self.sigma_period - 1, f'SJS{str(cable_number).zfill(2)}'].to_list()
            max_upside = max(upside)
            min_upside = min(upside)

            downside = day_df.loc[i:i + self.sigma_period - 1, f'SJX{str(cable_number).zfill(2)}'].to_list()
            max_downside = max(downside)
            min_downside = min(downside)

            sigma = np.log((max_downside - min_downside) / (max_upside - min_upside))
            sigma_list.append(sigma)

        return np.asarray(sigma_list)

    def get_cable_bic_set(self, cable_number, n_clusters=9, iterations=20):
        index = self.date_list.copy()
        index.append('mean')
        bic_df = pd.DataFrame(index=index, columns=[i+1 for i in range(n_clusters)])

        for date in self.date_list:
            bic_list, bic_error_list, best_number_of_clusters = self.get_bic_list(date, cable_number, n_clusters, iterations)
            bic_df.loc[date, :] = bic_list

        for cluster in range(n_clusters):
            cluster_bic = bic_df.iloc[:-1, cluster].to_list()
            bic_mean = sum(cluster_bic) / float(n_clusters)
            bic_df.loc['mean', cluster + 1] = bic_mean

        mean_bic_list = bic_df.loc['mean', :].to_list()
        best_cluster = mean_bic_list.index(min(mean_bic_list)) + 1
        return bic_df, best_cluster

    def get_bic_list(self, date, cable_number, n_clusters=9, iterations=20):
        sigma_list = self.get_sigma_list(date, cable_number)

        n_clusters = np.arange(1, n_clusters + 1)
        bic_list = []
        bic_error_list = []

        for n in n_clusters:
            bic = []
            for _ in range(iterations):
                sigma_np = np.expand_dims(sigma_list, 1)
                gmm = mixture.GaussianMixture(n, n_init=2).fit(sigma_np)
                bic.append(gmm.bic(sigma_np))

            val = np.mean(get_best_bic(np.array(bic), int(iterations / 5)))
            err = np.std(bic)
            bic_list.append(val)
            bic_error_list.append(err)

        best_number_of_clusters = get_number_of_clusters(bic_list)

        return bic_list, bic_error_list, best_number_of_clusters

    def get_gmm_model_with_n_clusters(self, date, cable_number, n_clusters, period=90):
        sigma_list = self.get_sigma_list(date, cable_number)
        model, gmm_x, gmm_y_sum = self.get_gmm_result(sigma_list, n_clusters, period)
        gmm_df = pd.DataFrame(columns=['x', 'y', 'mean'])
        gmm_df.loc[:, 'x'] = gmm_x
        gmm_df.loc[:, 'y'] = gmm_y_sum
        gmm_df.loc[:1, 'mean'] = model.means_
        return gmm_df

    def get_gmm_model(self, date, cable_number, n_clusters=9, period=90, iterations=20):
        sigma_list = self.get_sigma_list(date, cable_number)
        bic_list, bic_error_list, best_n_clusters = self.get_bic_list(date, cable_number, n_clusters, iterations)

        model, gmm_x, gmm_y_sum = self.get_gmm_result(sigma_list, best_n_clusters, period)
        return model, gmm_x, gmm_y_sum

    def get_gmm_result(self, sigma_list, best_n_clusters, period):
        model = mixture.GaussianMixture(n_components=best_n_clusters, covariance_type="full", tol=0.001)
        model = model.fit(np.expand_dims(sigma_list, 1))

        weights = get_weights(sigma_list)
        counts, bins, patches = plt.hist(sigma_list, weights=weights, label='Histogram', histtype='stepfilled',
                                         bins=period)
        plt.close()

        gmm_x = np.linspace(min(bins), max(bins), len(weights))
        gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
        for m, c, w in zip(model.means_.ravel(), model.covariances_.ravel(), model.weights_.ravel()):
            gauss = gauss_function(x=gmm_x, amp=1, x0=m, sigma=np.sqrt(c))
            gmm_y_sum += gauss / np.trapz(gauss, gmm_x) * w

        return model, gmm_x, gmm_y_sum

    def get_cable_gmm(self, cable_number):
        model_list = []
        gmm_x_list = []
        gmm_y_list = []

        for date in self.date_list:
            model, x, y_sum = self.get_gmm_model(date, cable_number)
            model_list.append(model)
            gmm_x_list.append(x)
            gmm_y_list.append(y_sum)

        return model_list, gmm_x_list, gmm_y_list

    def get_gmm_results_with_n_clusters(self, cable_number, n_cluster):
        gmm_dict = dict()
        x_min = 10000
        x_max = -0.00001

        for date in self.date_list:
            gmm_df = self.get_gmm_model_with_n_clusters(date, cable_number, n_cluster)
            x_list = gmm_df.loc[:, 'x'].to_list()
            if min(x_list) < x_min:
                x_min = min(x_list)
            if max(x_list) > x_max:
                x_max = max(x_list)
            gmm_dict.update({date: gmm_df})

        return gmm_dict, x_min, x_max

    def plot_sigma_list(self, date, cable_number, sigma_list, period=90, alpha=0.3):
        cable_name = f'SJ{(str(cable_number).zfill(2))}'

        # plot histogram
        fig, ax_hist = plt.subplots(ncols=1)
        counts, bins, patches = ax_hist.hist(sigma_list, weights=get_weights(sigma_list), label='Histogram',
                                             color='tab:blue', ec='tab:blue',
                                             histtype='stepfilled', alpha=alpha, bins=period)

        # ecdf
        ecdf_list = []
        ratio = 0
        for i in range(period):
            ratio += (counts[i] / sum(counts)) * 100
            ecdf_list.append(ratio)

        ecdf_x = np.linspace(min(bins), max(bins), len(ecdf_list))

        # plot ecdf
        ax_ecdf = ax_hist.twinx()
        sub_ecdf = ax_ecdf.plot(ecdf_x, ecdf_list, label='ECDF')

        # legend
        ax_sum = patches + sub_ecdf
        labels = [ax.get_label() for ax in ax_sum]

        ax_hist.legend(ax_sum, labels, loc=0)
        ax_hist.set(ylabel='Probability Density (%)', xlabel=u'\u03B6')
        ax_ecdf.set(ylabel='Cumulative Distribution (%)')
        plt.suptitle(f'{date}, {cable_name}')
        plt.show()

    def plot_bic_error(self, date, cable_number, n_clusters=9, iterations=20, fontsize=10):
        bic_list, bic_error_list, best_n_clusters = self.get_bic_list(date, cable_number, n_clusters, iterations)
        cable_name = f'SJ{(str(cable_number).zfill(2))}'
        n_clusters = np.arange(1, n_clusters + 1)

        plt.rcParams.update({'font.size': fontsize})
        plt.errorbar(n_clusters, bic_list, yerr=bic_error_list, label='BIC')
        plt.title(f'BIC Scores, ({date}, {cable_name})')
        plt.xticks(n_clusters)
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        plt.legend()

    def plot_gmm(self, date, cable_number, gmm_x, gmm_y_sum, period=90, fontsize=10, alpha=0.2, linewidth=2):
        # histogram
        sigma_list = self.get_sigma_list(date, cable_number)

        # plot histogram
        fig, ax_hist = plt.subplots(ncols=1)
        counts, bins, patches = ax_hist.hist(sigma_list, weights=get_weights(sigma_list), label='Histogram',
                                             color='tab:blue', ec='tab:blue',
                                             histtype='stepfilled', alpha=alpha, bins=period)
        # plot gmm
        ax_gmm = ax_hist.twinx()
        ax_gmm = ax_gmm.plot(gmm_x, gmm_y_sum, color='black', lw=linewidth, label='Fitted GMM')

        # legend
        ax_sum = patches + ax_gmm
        labels = [ax.get_label() for ax in ax_sum]

        plt.rcParams.update({'font.size': fontsize})
        ax_hist.legend(ax_sum, labels, loc=0)
        ax_hist.set(ylabel='Probability Density (%)', xlabel=u'\u03B6')
        plt.suptitle(f'{date}, SJ{(str(cable_number).zfill(2))}')
        plt.show()

    def plot_contour(self, gmm_dict, x_min, x_max, cable_number, n_xp=10000):
        xp = np.linspace(x_min - 0.01, x_max + 0.01, n_xp)
        yp = self.date_list
        zp = np.empty((len(yp), len(xp)))

        for i, date in enumerate(self.date_list):
            gmm_df = gmm_dict[date]
            x_list = gmm_df.loc[:, 'x'].to_list()
            x_index_list = map_x_axis(x_list, xp)

            y_list = gmm_df.loc[:, 'y'].to_list()

            for j, x_index in enumerate(x_index_list):
                zp[i][x_index] = y_list[j]

        contours = plt.contour(xp, yp, zp, cmap='turbo')
        norm = mcolors.Normalize(vmin=contours.cvalues.min(), vmax=contours.cvalues.max())

        sm = plt.cm.ScalarMappable(norm=norm, cmap=contours.cmap)
        sm.set_array([])

        title = f'PDF contour of SJ{str(cable_number).zfill(2)}'
        if self.hourly:
            title = f'{title}, hour {self.hour}'

        plt.colorbar(sm, ticks=contours.levels)
        plt.title(title)
        plt.xlabel(u'\u03B6')
        plt.show()


def get_dup_cable_numbers(tension_df):
    cable_list = tension_df.columns.to_list()
    cable_numbers = []

    for cable in cable_list:
        cable_number = int(cable[-2:])
        cable_numbers.append(cable_number)

    cable_numbers = [x for i, x in enumerate(cable_numbers) if i != cable_numbers.index(x)]
    return cable_numbers


def get_best_bic(array, length):
    indices = np.argsort(array)[:length]
    return array[indices]


def get_number_of_clusters(bic_list):
    return bic_list.index(min(bic_list)) + 1


def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


def get_weights(array):
    return (np.ones(len(array)) / len(array)) * 100


def map_x_axis(old_axis, new_axis):
    new_min = min(new_axis)
    new_max = max(new_axis)
    new_len = len(new_axis)
    one_step = (new_max - new_min) / new_len

    index_list = []
    for old_value in old_axis:
        index = int((old_value - new_min) / one_step)
        index_list.append(index)

    return index_list


if __name__ == '__main__':
    gmm = CableGMM()
    print(gmm.data_path)
