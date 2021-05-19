from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_single_cable(tension_df, cable_name, date=datetime(2000, 1, 1), start=0, end=100,
                      figsize=(80, 15), fontsize=60, linewidth=3):
    x_value = [date + timedelta(seconds=(start + 90) * 0.5) + timedelta(seconds=i * 0.5) for i in range(end - start)]
    x_format = mdates.DateFormatter('%H:%M')

    y_value = tension_df.loc[start:end-1, cable_name].to_list()

    mpl.rcParams['lines.linewidth'] = linewidth
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(x_value, y_value)
    ax.set(xlabel='Time', ylabel='Cable Tension (kN)')
    ax.xaxis.set_major_formatter(x_format)
    ax.title.set_text(f'Cable tension of {cable_name}')

    plt.show()


def plot_up_down_cables(tension_df, cable_number, date=datetime(2020, 1, 1), start=0, end=100,
                        figsize=(80, 15), fontsize=60, linewidth=3):
    x_value = [date + timedelta(seconds=(90 + 101) * 0.5) + timedelta(seconds=(i + (90 + 101)) * 0.5) for i in range(end - start)]
    x_format = mdates.DateFormatter('%H:%M')

    cable_number = str(cable_number).zfill(2)
    y_values = [tension_df.loc[start:end - 1, f'SJ{i}{cable_number}'].to_list() for i in ['S', 'X']]

    mpl.rcParams['lines.linewidth'] = linewidth
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i, name in enumerate(['S', 'X']):
        ax.plot(x_value, y_values[i], label=f'SJ{name}{cable_number}')
    ax.set(xlabel='Time', ylabel='Cable Tension (kN)')
    ax.xaxis.set_major_formatter(x_format)
    ax.title.set_text(f'Cable tensions of SJ{cable_number}')

    plt.legend()
    plt.show()


def scatter_up_down_cables(tension_df, cable_number, start=0, end=100,
                           figsize=(80, 15), s=10, fontsize=60):
    cable_number = str(cable_number).zfill(2)
    x = tension_df.loc[start:end - 1, f'SJS{cable_number}'].to_list()
    y = tension_df.loc[start:end - 1, f'SJX{cable_number}'].to_list()

    plt.rcParams.update({'font.size': fontsize})

    plt.figure(figsize=figsize)
    plt.xlim([-80, 365])
    plt.ylim([-80, 365])
    plt.scatter(x, y, s=s, c='#000000')
    plt.title(f'Cable Tensions of SJ{cable_number}')
    plt.xlabel(f'SJS{cable_number} (kN)')
    plt.ylabel(f'SJX{cable_number} (kN)')
    plt.show()


def plot_single_stl(stl, date, cable_name, start, end,
                    fig_height=4, fontsize=60, linewidth=5, pad=1.5):
    date = datetime.strptime(date, '%Y-%m-%d')
    x_value = [date + timedelta(seconds=i * 0.5) + timedelta(seconds=(start + 90) * 0.5) for i in range(end - start)]
    x_format = mdates.DateFormatter('%H:%M')

    plt.rcParams.update({'font.size': fontsize})

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(80, 15 * fig_height))

    # plt.suptitle("%s, %s" % (date.strftime("%Y-%m-%d"), cable_name))
    fig.tight_layout(pad=pad)
    for i, ax in enumerate(axs):
        ax.plot(x_value, stl.iloc[start:end, i], linewidth=linewidth)
        # ax.set(ylabel=stl.iloc[start:end, i].name)
        ax.xaxis.set_major_formatter(x_format)

    plt.show()


def plot_cable_stl_by_type(stl, date_list, cable_name, stl_type,
                           start, end, fontsize=60, linewidth=5, title=False):
    date = datetime.strptime(date_list[0], '%Y-%m-%d')
    x_value = [date + timedelta(seconds=i * 0.5) + timedelta(seconds=(start + 90) * 0.5) for i in range(end - start)]
    x_format = mdates.DateFormatter('%H:%M')

    plt.rcParams.update({'font.size': fontsize})

    plt.figure(figsize=(80, 15))
    for date in date_list:
        plt.plot(x_value, stl.iloc[start:end, stl_type.value], label=date, linewidth=linewidth)

    plt.gca().xaxis.set_major_formatter(x_format)
    if title is True:
        plt.suptitle(f'{cable_name}')
    # plt.legend(loc=2)
    plt.show()


def plot_stl_cross_correlation(cross_correlate_df, cable_name,
                           fontsize=60, figsize=(80, 15), linewidth=5, loc=3):
    x_value = cross_correlate_df.index.to_list()

    plt.rcParams.update({'font.size': fontsize})

    plt.figure(figsize=(figsize))
    for column in cross_correlate_df.columns:
        plt.plot(x_value, cross_correlate_df.loc[:, column], linewidth=linewidth, label=column)

    plt.suptitle(f'{cable_name}, cross correlation')
    plt.legend(loc=loc)
    plt.show()


def plot_stl_cross_correlation_by_type(cross_correlate_df, adjacency_type,
                                       fontsize=60, figsize=(80, 15), linewidth=5):
    x_value = cross_correlate_df.columns.tolist()

    plt.rcParams.update({'font.size': fontsize})

    plt.figure(figsize=(figsize))
    for cable in cross_correlate_df.index.to_list():
        plt.plot(x_value, cross_correlate_df.loc[cable, :], linewidth=linewidth, label=cable)

    plt.suptitle(f'{adjacency_type.name}, cross correlation')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.show()
