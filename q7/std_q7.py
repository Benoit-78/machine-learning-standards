# -*- coding: utf-8 -*-
"""
Author: Benoit DELORME
Mail: delormebenoit211@gmail.com
Creation date: 26th June 2021
Main objective: provide an IT version of the tools of quality, as described
                by Dr. Ishikawa in its book 'Guide for Quality Control, 1968'
"""

import statistics as stat

from collections import Counter

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from matplotlib.patches import Ellipse



class QualityTool():
    """
    Quality tools are used in very different situations.
    However, their purposes are often similar:
    - analysis,
    - control,
    - decision taking,
    - ...
    These common purposes are gathered in the present class.
    """
    def __init__(self, purpose=None, scale=None, process=None, line=None,
        product=None, datetime=None, multiple='Unique'):
        self.purpose = purpose
        self.scale = scale
        self.process = process
        self.line = line
        self.product = product
        self.datetime = datetime
        self.multiple = multiple
        sns.set_theme(style="darkgrid")

    def general_info(self):
        """
        Displays a table of general information below the quality tool.
        """
        temp_df = pd.DataFrame(
            columns=['Value'],
            index=['Purpose', 'Scale', 'Process', 'Line',
                   'Product', 'Date', 'Shift', 'Divers'],
            data=[self.purpose, self.scale, self.process, self.line,
                  self.product, self.datetime])
        return temp_df



# 2nd tool
class Histogram(QualityTool):
    """
    What is a histogram?
        -> a histogram is a graph that represents the distribution of a
        quantitative variable.
    What use is a histogram?
        -> shows the global and local tendencies, the outliers, ...
    """
    def __init__(self, df, feature, quantile_sup=1, quantile_inf=0):
        super().__init__()
        self.feature = feature
        self.data = df[feature]
        self.quantile_sup = quantile_sup
        self.quantile_inf = quantile_inf

    def clean_data(self):
        """
        Prepare the data to be ready for analysis.
        """
        def remove_currency_symbols():
            for currency in ['€', '$', '£']:
                self.data = [element.replace(currency, '')
                        for element in self.data
                        if currency in element]

        def replace_empty_strings():
            self.data = [element.replace(' ', '') for element in self.data]

        def replace_comma_by_points():
            self.data = [element.replace(',', '.') for element in self.data]

        def make_floats():
            self.data = [float(element) for element in self.data]

        if self.data.dtype in ['object', 'str']:
            remove_currency_symbols()
            replace_empty_strings()
            replace_comma_by_points()
            make_floats()
        return pd.Series(self.data)

    def data_without_outliers(self):
        """
        Remove outliers according to specification.
        """
        if self.quantile_sup != 1:
            self.data = self.data[self.data < self.data.quantile(self.quantile_sup)]
        if self.quantile_inf != 0:
            self.data = self.data[self.data > self.data.quantile(self.quantile_inf)]
        return self.data

    def statistical_metadata(self):
        """
        Return general purpose statistical metrics.
        """
        mean = stat.mean(self.data)
        sigma = stat.stdev(self.data)
        skewness = ss.skew(self.data)
        kurtosis = ss.kurtosis(self.data)
        indic_list = [mean, sigma, skewness, kurtosis]
        indic_list = [round(number, 2) for number in indic_list]
        return indic_list

    def plot(self):
        """
        Plot the histogram.
        """
        data = self.clean_data()
        data = self.data_without_outliers()
        [mean, sigma, skewness, kurtosis] = self.statistical_metadata()
        # plt.tight_layout(5.0)
        fig, (ax_box, ax_dist) = plt.subplots(2, 1, sharex='all',
                                              gridspec_kw={
                                                  'height_ratios':[1, 5]
                                                  })
        fig.suptitle('Distribution of ' + self.feature)
        sns.set_theme(style="darkgrid")
        sns.boxplot(data=data, ax=ax_box)
        ax_box.set_xlabel('')
        #
        sns.histplot(data, kde=True, ax=ax_dist, bins=60,
                     label='\u03C3 = ' + str(sigma) + '\n' +
                            'skew = ' + str(skewness) + '\n' +
                            'kurtosis = ' + str(kurtosis))
        ax_dist.set_xlabel('Values')
        ax_dist.set_ylabel('Amount')
        ax_dist.axvline(x=stat.mean(data), c='k', linestyle='--',
                        label='\u03BC = ' + str(mean))
        ax_dist.legend(loc='best')
        return super().general_info()



# 3rd tool
class ControlChart(QualityTool):
    """
    A control chart is a QC tool designed to:
    - reveal the state of control of a process,
    - identify the outliers,
    - warn the user when the process becomes out of control,
    - help decisions and actions.
    """
    def __init__(self, data, feature_name, reverse=False):
        super().__init__()
        self.data = data
        self.feature_name = feature_name
        if not reverse:
            self.orientation = 'line'
        else:
            self.orientation = 'column'

    def array_compute(self):
        """
        Given the data, computes the lists of values that will be displayed
        on the graph.
        """
        n_samples = self.data.shape[0]
        # AVERAGES
        x_bar_list = [round(element, 1) for element in self.data.mean(axis=0)]
        x_bar_bar = [stat.mean(x_bar_list)] * len(x_bar_list)
        # RANGES
        max_list = [round(element, 1) for element in self.data.max(axis=0)]
        min_list = [round(element, 1) for element in self.data.min(axis=0)]
        r_list = []
        zip_object = zip(max_list, min_list)
        for max_list, min_list in zip_object:
            r_list.append(max_list - min_list)
        r_bar = [stat.mean(r_list)] * len(r_list)
        # Control limits
        constants_df = pd.read_csv(r'.\3_constants.csv')
        constants_df.set_index('sample size', inplace=True)
        a_2 = float(constants_df.loc[n_samples]['A2'])
        d_3 = float(constants_df.loc[n_samples]['D3'])
        d_4 = float(constants_df.loc[n_samples]['D4'])
        # Averages
        x_bar_ucl = [x_bar_bar[0] + r_bar[0] * a_2] * len(r_list)
        x_bar_lcl = [x_bar_bar[0] - r_bar[0] * a_2] * len(r_list)
        # Ranges
        r_ucl = [r_bar[0] * d_4] * len(r_list)
        r_lcl = [r_bar[0] * d_3] * len(r_list)
        return [x_bar_list, x_bar_bar,
                r_list, r_bar,
                x_bar_ucl, x_bar_lcl,
                r_ucl, r_lcl]

    def outliers_coordinates(self, values, ucl, lcl):
        """
        Given the data and the control limits, identify the points outside the
        latest.
        """
        outliers_coord = []
        for i, value in enumerate(values):
            if value > ucl or value < lcl:
                outlier_x = i
                outlier_y = values[i]
                outliers_coord.append((outlier_x, outlier_y))
        return outliers_coord

    def outliers_circles(self, coord_x, coord_y, control_range, length):
        """
        Display circles around outliers to make them visible for the user.
        """
        ellipse = Ellipse((coord_x, coord_y),
                          length / 12,
                          control_range / 2,
                          edgecolor='red',
                          fill=False)
        return ellipse

    def average_ylim(self, ucl, lcl):
        """
        Given a list of values, determines which y limits are appropriate
        to have a relevant plot.
        """
        control_range = ucl - lcl
        ylim_max = ucl + control_range
        ylim_min = lcl - control_range
        return (ylim_min, ylim_max)

    def range_ylim(self, ucl, lcl):
        """
        Determines the optimal vertical frame.
        """
        control_range = ucl - lcl
        ylim_max = 3 * control_range + - control_range / 10
        ylim_min = 0 - control_range / 10
        return (ylim_min, ylim_max)

    def plot_array(self):
        """
        Display the control chart, with values, averages, ranges,
        control limits, ...
        """
        [x_bar_list, x_bar_bar,
         r_list, r_bar,
         x_bar_ucl, x_bar_lcl,
         r_ucl, r_lcl] = self.array_compute()
        # Set the size
        data_len = len(x_bar_list)
        fig, _ = plt.subplots(ncols=2, nrows=2, sharex='col',
                                figsize=(math.sqrt(data_len * 5),
                                         math.sqrt(data_len * 3)))
        grid = plt.GridSpec(2, 6, wspace=0.3, hspace=0.2)
        ave_chart = plt.subplot(grid[0, :4])
        ave_hist = plt.subplot(grid[0, 4:])
        ran_chart = plt.subplot(grid[1, :4])
        ran_hist = plt.subplot(grid[1, 4:])
        fig.suptitle('Control chart: {}'.format(self.feature_name))
        # Set same xlim and ylim for the corresponding plots
        ave_ylim = (self.average_ylim(x_bar_ucl[0], x_bar_lcl[0]))
        ran_ylim = (self.range_ylim(r_ucl[0], r_lcl[0]))
        hist_xlim = (0, data_len)
        # AVERAGES CHART
        ave_chart.set_title('Averages')
        ave_chart.set_ylim(ave_ylim)
        major_ticks = np.arange(0, 20, 5)
        ave_chart.grid(axis='x')
        ave_chart.set_xticks(major_ticks)
        ave_chart.grid(axis='x')
        ave_chart.plot(x_bar_list, 'o-', color='k', markerfacecolor='orange',
                       markersize=8, linewidth=1)
        ave_chart.plot(x_bar_bar, color='k')
        # UCL and LCL
        ave_chart.plot(x_bar_ucl, '-', color='orange', linewidth=1)
        ave_chart.text(x=len(x_bar_list) - 2, y=x_bar_ucl[0] + 1, s='UCL')
        ave_chart.plot(x_bar_lcl, '-', color='orange', linewidth=1)
        ave_chart.text(x=len(x_bar_list) - 2, y=x_bar_lcl[0] - 2, s='LCL')
        # Outliers
        x_outliers_coord = self.outliers_coordinates(x_bar_list,
                                                     x_bar_ucl[0],
                                                     x_bar_lcl[0])
        control_range = x_bar_ucl[0] - x_bar_lcl[0]
        for outlier in x_outliers_coord:
            ave_chart.add_artist(self.outliers_circles(
                outlier[0], outlier[1], control_range, data_len))
        # AVERAGES HISTOGRAM
        ave_hist.set_xlim(hist_xlim)
        ave_hist.set_ylim(ave_ylim)
        ave_hist.grid(axis='x')
        ave_hist.grid(axis='y')
        ave_hist.get_xaxis().set_visible(True)
        ave_hist.get_yaxis().set_visible(True)
        ave_hist.grid(axis='x')
        ave_hist.grid(axis='y')
        ave_hist.hist(x_bar_list, orientation='horizontal',
                      color='orange', edgecolor='k', bins=10)
        # RANGES CHART
        ran_chart.set_title('Ranges')
        ran_chart.set_ylim(ran_ylim)
        ran_chart.grid(axis='x')
        ran_chart.set_xticks(major_ticks)
        ran_chart.grid(axis='x')
        ran_chart.plot(r_list, 'D-', color='k', markerfacecolor='c',
                       markersize=8, linewidth=1)
        ran_chart.plot(r_bar, color='k')
        # UCL and LCL
        ran_chart.plot(r_ucl, '-', color='c', linewidth=1)
        ran_chart.text(x=len(r_list) - 2, y=r_ucl[0] + 2, s='UCL')
        ran_chart.plot(r_lcl, '-', color='c', linewidth=1)
        ran_chart.text(x=len(r_list) - 2, y=r_lcl[0] - 5, s='LCL')
        # Outliers
        r_outliers_coord = self.outliers_coordinates(r_list,
                                                     r_ucl[0],
                                                     r_lcl[0])
        for outlier in r_outliers_coord:
            ran_chart.add_artist(
                self.outliers_circles(
                    outlier[0], outlier[1], r_ucl[0] - r_lcl[0], data_len))
        # RANGES HISTOGRAM
        ran_hist.set_xlim(hist_xlim)
        ran_hist.set_ylim(ran_ylim)
        ran_hist.grid(axis='x')
        ran_hist.grid(axis='y')
        ran_hist.get_xaxis().set_visible(True)
        ran_hist.get_yaxis().set_visible(True)
        ran_hist.grid(axis='x')
        ran_hist.grid(axis='y')
        ran_hist.hist(r_list, orientation='horizontal',
                      color='c', edgecolor='k', bins=10)
        _ = ''  # to prevent a perturbative error message
        # return _

    def control_chart_bivariate(self, original_df, x_feature, y_feature,
                                group_by):
        """
        Control chart with y_feature as values & x_feature as dates.
        """
        dataframe = original_df[[x_feature, y_feature]]
        dataframe[x_feature] = pd.to_datetime(dataframe[x_feature])
        if group_by == 'week':
            dataframe[x_feature] = [element.isocalendar()[1]
                             for element in dataframe[x_feature]]
        if group_by == 'month':
            dataframe[x_feature] = dataframe[x_feature].dt.month
        dataframe = dataframe.groupby([x_feature]).mean().reset_index()
        # Average line and control lines
        feature_mean = dataframe[y_feature].mean()
        dataframe['feature_mean'] = [feature_mean] * dataframe.shape[0]
        feature_stdev = dataframe[y_feature].std()
        dataframe['feature_ucl'] = dataframe['feature_mean'] + 3*feature_stdev
        dataframe['feature_lcl'] = dataframe['feature_mean'] - 3*feature_stdev
        # Plot
        _, _ = plt.subplots(figsize=(8, 8))
        plt.title(y_feature)
        plt.xlabel('Semaines calendaires')
        plt.ylabel(y_feature)
        plt.ylim(min(dataframe[y_feature]) * 1.1,
                 max(dataframe[y_feature]) * 1.1)
        _ = plt.plot(dataframe[x_feature], dataframe[y_feature],
                     'c', linewidth=1, marker='o', color='k',
                     markerfacecolor='orange', markersize=8)
        _ = plt.plot(dataframe[x_feature], dataframe['feature_mean'],
                     'k', linewidth=3)
        _ = plt.plot(dataframe[x_feature], dataframe['feature_ucl'],
                     '--', color='k')
        _ = plt.plot(dataframe[x_feature], dataframe['feature_lcl'],
                     '--', color='k')
        plt.show()



class SpecialCauseDetector():
    """
    Provide identifiers of special causes in the given data.
    """
    def __init__(self, data, feature_name):
        self.data = data
        self.feature_name = feature_name
        [self.values, self.mean,
         _, _,
         self.ucl, self.lcl,
         _, _] = ControlChart(data, feature_name).array_compute()
        self.mean = self.mean[0]
        self.ucl = self.ucl[0]
        self.lcl = self.lcl[0]
        self.no_warning = 'No warning'

    def one_side_sequence(self, n_points=7):
        """
        Identify a sequence of points all above (or all below) the average
        value.
        """
        indicators_list = []
        for i, value in enumerate(self.values):
            if value > self.mean:
                indicators_list.append(1)
            else:
                indicators_list.append(0)
        # Transforming the list in string for comparison
        zeros_string = ''.join([str(element) for element in [0] * n_points])
        ones_string = ''.join([str(element) for element in [1] * n_points])
        indic_string = ''.join([str(element) for element in indicators_list])
        # Check the presence of such a sequence in the values
        if zeros_string in indic_string:
            i = len(indic_string.split(zeros_string)[0]) + 1
            result = 'Warning: sequence of {} or more successive points under'\
                ' the mean value. Starting from point {}.'.format(n_points, i)
        elif ones_string in indic_string:
            i = len(indic_string.split(ones_string)[0]) + 1
            result = 'Warning: sequence of {} or more successive points under'\
                ' the mean value. Starting from point {}.'.format(n_points, i)
        else:
            result = self.no_warning
        return result

    def monotone_sequence(self, n_points=6):
        """
        Identify a sequence of points that is strictly ascending (or strictly
        descending)
        """
        indicators_list = []
        for i in range(1, len(self.values)):
            if self.values[i] > self.values[i - 1]:
                indicators_list.append(1)
            else:
                indicators_list.append(0)
        # Transforming the list in string for comparison
        zeros_string = ''.join([str(element) for element in [0] * n_points])
        ones_string = ''.join([str(element) for element in [1] * n_points])
        indic_string = ''.join([str(element) for element in indicators_list])
        # Check the presence of such a sequence in the values
        if zeros_string in indic_string:
            i = len(indic_string.split(zeros_string)[0]) + 1
            result = 'Warning: monotone sequence of {} or more successive'\
                ' points. Starting from point {}.'.format(
                n_points, i)
        elif ones_string in indic_string:
            i = len(indic_string.split(ones_string)[0]) + 1
            result = 'Warning: monotone sequence of {} or more successive'\
                ' points. Starting from point {}.'.format(
                n_points, i)
        else:
            result = self.no_warning
        return result

    def wobble_sequence(self, n_points=14):
        """
        Identify a sequence of points that oscilate around the average value
        (one above and one below sequentially)
        """
        indicators_list = []
        for i in range(1, len(self.values)):
            if self.values[i] > self.values[i - 1]:
                indicators_list.append(1)
            else:
                indicators_list.append(0)
        # Transforming the list in string for comparison
        wobble_string = ''.join([str(element)
                                 for element in ['01'] * int(n_points / 2)])
        indic_string = ''.join([str(element)
                                for element in indicators_list])
        # Check the presence of such a sequence in the values
        if wobble_string in indic_string:
            i = len(indic_string.split(wobble_string)[0]) + 1
            result = 'Warning: wobble sequence of {} or more successive points.'\
                ' Starting from point {}.'.format(n_points, i)
        else:
            result = self.no_warning
        return result

    def two_sigma_sequence(self):
        """
        Identify a sequence of 3 points not in the central 2-sigma zone,
        all above or all below average value.
        """
        indic_list = []
        for value in self.values:
            cond_1 = (value - self.mean > (self.ucl - self.lcl) / 3)
            cond_2 = (value < self.ucl)
            cond_3 = (value - self.mean < -(self.ucl - self.lcl) / 3)
            cond_4 = (value > self.lcl)
            # Partie sup
            if cond_1 and cond_2:
                indic_list.append(1)
            # Partie inf
            if cond_3 and cond_4:
                indic_list.append(-1)
            else:
                indic_list.append(0)
        # Transforming the list in string for comparison
        str_1 = '101'
        str_2 = '11'
        str_3 = '-10-1'
        str_4 = '-1-1'
        indic_string = ''.join([str(element) for element in indic_list])
        # Check the presence of such a sequence in the values
        if str_1 in indic_string:
            i = len(indic_string.split(str_1)[0]) + 1
            result = 'Warning: two neighbour points above upper 2-sigma'\
                ' limit. Starting from point {}.'.format(i)
        if str_2 in indic_string:
            i = len(indic_string.split(str_2)[0]) + 1
            result = 'Warning: two successive points above upper 2-sigma'\
                ' limit. Starting from point {}.'.format(i)
        if str_3 in indic_string:
            i = len(indic_string.split(str_3)[0]) + 1
            result = 'Warning: two neighbour points under lower 2-sigma'\
                ' limit. Starting from point {}.'.format(i)
        if str_4 in indic_string:
            i = len(indic_string.split(str_4)[0]) + 1
            result = 'Warning: two successive points under lower 2-sigma'\
                ' limit. Starting from point {}.'.format(i)
        else:
            result = self.no_warning
        return result

    def one_sigma_sequence(self):
        """
        Identify a sequence of 4 points not in the central one-sigma zone,
        all above or all below average value.
        """
        indic_list = []
        for value in self.values:
            cond_1 = (value - self.mean > (self.ucl - self.lcl) / 6)
            cond_2 = (value < self.ucl)
            cond_3 = (value - self.mean < -(self.ucl - self.lcl) / 6)
            cond_4 = (value > self.lcl)
            # Partie sup
            if cond_1 and cond_2:
                indic_list.append(1)
            # Partie inf
            if cond_3 and cond_4:
                indic_list.append(-1)
            else:
                indic_list.append(0)
        # Transforming the list in string for comparison
        up_list = ['01111',
                   '10111',
                   '11011',
                   '11101',
                   '11110']
        low_list = ['0-1-1-1-1',
                    '-10-1-1-1',
                    '-1-10-1-1',
                    '-1-1-10-1',
                    '-1-1-1-10']
        indic_string = ''.join([str(element) for element in indic_list])
        # Check the presence of such a sequence in the values
        for string in up_list:
            if string in indic_string:
                i = len(indic_string.split(string)[0]) + 1
                result = 'Warning: 4 neighbour points above upper'\
                    '1-sigma limit. Starting from point {}.'.format(i)
            else:
                result = self.no_warning
        for string in low_list:
            if string in indic_string:
                i = len(indic_string.split(string)[0]) + 1
                result = 'Warning: 4 neighbour points under lower'\
                    '1-sigma limit. Starting from point {}.'.format(i)
            else:
                result = self.no_warning
        return result

    def central_sequence(self, n_points=15):
        """
        Identify a sequence of 15 points in the central one-sigma zone.
        """
        indic_list = []
        for value in self.values:
            if abs(value - self.mean) < (self.ucl - self.lcl) / 6:
                indic_list.append(1)
            else:
                indic_list.append(0)
        # Transforming the list in string for comparison
        ones_string = ''.join([str(element) for element in [1] * n_points])
        indic_string = ''.join([str(element) for element in indic_list])
        # Check the presence of such a sequence in the values
        if ones_string in indic_string:
            i = len(indic_string.split(ones_string)[0]) + 1
            result = 'Warning: central sequence of {} or more successive '\
                'points. Starting from point {}.'.format(n_points, i)
        else:
            result = self.no_warning
        return result

    def sides_only_sequence(self, n_points=8):
        """
        Identify a sequence of 8 points not in the central one-sigma zone.
        """
        indic_list = []
        for value in self.values:
            if abs(value - self.mean) > (self.ucl - self.lcl) / 6:
                indic_list.append(1)
            else:
                indic_list.append(0)
        # Transforming the list in string for comparison
        ones_string = ''.join([str(element) for element in [1] * n_points])
        indic_string = ''.join([str(element) for element in indic_list])
        # Check the presence of such a sequence in the values
        if ones_string in indic_string:
            i = len(indic_string.split(ones_string)[0]) + 1
            result = 'Warning: sides_only sequence of {} or more successive '\
                'points. Starting from point {}.'.format(n_points, i)
        else:
            result = self.no_warning
        return result

    def report(self):
        """
        Return a dataframe of principal indicators.
        """
        report_df = pd.DataFrame(columns=['Comment'])
        report_df.loc['One-side sequence'] = [self.one_side_sequence()]
        report_df.loc['Monotone sequence'] = [self.monotone_sequence()]
        report_df.loc['Wobble sequence'] = [self.wobble_sequence()]
        report_df.loc['2 sigma sequence'] = [self.two_sigma_sequence()]
        report_df.loc['1 sigma sequence'] = [self.one_sigma_sequence()]
        report_df.loc['Central sequence'] = [self.central_sequence()]
        report_df.loc['Sides only sequence'] = [self.sides_only_sequence()]
        return report_df



# 4th tool
class Pareto(QualityTool):
    """
    What is a Pareto graph?
    -> A bar graph of the most common elements of the list, sorted according
    their frequency.
    What use is a Pareto graph?
    -> Identify the most common elements in a set and provide a basis for
    prioritization of action.
    """
    def __init__(self, dataframe, column, proportion_acceptance):
        super().__init__()
        self.dataframe = dataframe
        self.column = column
        self.rate = proportion_acceptance

    def categories_and_frequencies(self):
        """
        Return categories and their respective frequencies.
        Filter so that rare categories do not pollute the Pareto graph.
        """
        def raw_categories_and_frequencies():
            """Returns categories and their occurrence frequencies"""
            cat_counter = self.dataframe[self.column].value_counts(normalize=True)
            categories = list(cat_counter.index)
            proportions = list(cat_counter)
            return categories, proportions

        def df_replace_nan(categories, frequencies):
            new_df = pd.DataFrame({'Categories':categories,
                                   'Frequencies':frequencies})
            new_df = new_df.replace(np.nan, 'NaN')
            return new_df

        def first_categories_and_frequencies(cat_df):
            filtered_cat_df = cat_df[cat_df['Frequencies'] > self.rate]
            categories = list(filtered_cat_df['Categories'])
            frequencies = list(filtered_cat_df['Frequencies'])
            return categories, frequencies

        categories, frequencies = raw_categories_and_frequencies()
        categories_df = df_replace_nan(categories, frequencies)
        [categories,
         frequencies] = first_categories_and_frequencies(categories_df)
        frequencies = [frequency * 100 for frequency in frequencies]
        return [categories, frequencies]

    def eighty_pct_line(self, cum_frequencies_):
        """
        Find the point on the cumulative sum curb that has a 80% ordinate
        """
        [eighty_percent, x_inf, y_inf, y_sup] = [None] * 4
        for i, cum_frequency in enumerate(cum_frequencies_):
            if cum_frequency > 80 and i > 0:
                eighty_percent = True
                x_inf = i-1
                y_inf = cum_frequencies_[i-1]
                y_sup = cum_frequency
                break
        return [eighty_percent, x_inf, y_inf, y_sup]

    def eighty_pct_3_points(self, x_inf, y_inf, y_sup, frequencies_):
        """
        Abscissa of the 80% points
        """
        x_80 = x_inf + (80 - y_inf) / (y_sup - y_inf)
        abs_80 = [x_80, 0]
        pt_80 = [x_80, 80]
        ord_80 = [len(frequencies_)-1, 80]
        return [abs_80, pt_80, ord_80]

    def colors(self, cumulated_frequencies):
        """
        Affect a color to each category, whether it belongs to the 80% or not.
        """
        colors_list = ['orange']
        for frequency in cumulated_frequencies:
            if frequency<=80:
                colors_list.append('orange')
            else:
                colors_list.append('grey')
        # Adapt the shift by one
        colors_list.pop()
        return colors_list

    def plot(self):
        """
        Plot Pareto graph.
        """
        sns.set_theme(style="darkgrid")
        [categories, frequencies] = self.categories_and_frequencies()
        cum_frequencies = list(np.array(frequencies).cumsum())
        factor = len(categories) - 1
        fig = plt.figure(figsize=(math.sqrt(factor*5),
                                  math.sqrt(factor*3)))
        # Bins
        ax1 = fig.add_subplot(111)
        ax1.grid(False)
        plt.title(
            'Pareto diagram, feature \'{}\', proportion acceptance {}%'.format(
                self.column, self.rate))
        plt.xlabel('Categories')
        plt.xticks(rotation=45, ha='right')
        plt.ylim((0, 110))
        plt.ylabel('Occurrence in % of total')
        ax1.bar(categories, frequencies,
                color=self.colors(cum_frequencies), edgecolor='k', alpha=0.5)
        # Cumsum line
        ax2 = ax1.twinx()
        plt.ylim((0, 110))
        ax2.plot(range(0, len(cum_frequencies)),
                 cum_frequencies, color='red')
        # 80%
        eighty_percent = None
        [eighty_percent,
         x_inf,
         y_inf,
         y_sup] = self.eighty_pct_line(cum_frequencies)
        if eighty_percent and y_inf < 80:
            [abs_80, pt_80, ord_80] = self.eighty_pct_3_points(x_inf,
                                                               y_inf,
                                                               y_sup,
                                                               frequencies)
            # Vertical line
            ax3 = ax2.twinx()
            plt.ylim((0, 110))
            ax3.plot([abs_80[0], pt_80[0]],
                     [abs_80[1], pt_80[1]],
                     '--', color='red', linewidth=1)
            # Horizontal line
            ax4 = ax2.twinx()
            plt.ylim((0, 110))
            ax4.plot([pt_80[0], ord_80[0]],
                     [pt_80[1], ord_80[1]],
                     '--', color='red', linewidth=1)
        # Vertical lines
        plt.tick_params(axis='x', length=0)
        xcoords = [0, factor*1/5, factor*2/5, factor*3/5, factor*4/5, factor]
        for xcoord in xcoords:
            plt.axvline(x=xcoord, color='w')
        return super().general_info()



class PieChart(QualityTool):
    """
    What is a piechart?
        -> a piechart is a graph that represents the repartition of several
        categories in a feature.
        What use is a piechart?
        -> a piechart is used for qualitative features with few categories.
        For features with 4 categories or less, a piechart is used.
        For features with 5 categories or more, a Pareto graph is used.
    """
    def __init__(self, dataframe, column, rate):
        super().__init__()
        self.dataframe = dataframe
        self.column = column
        self.rate = rate

    def filter(self):
        """
        Reject the values representing less than rate % of the given feature
        length.
        """
        categories_dict = dict(Counter(self.dataframe[self.column]))
        df_length = self.dataframe.shape[0]
        new_dict = {category: count
                    for category, count in categories_dict.items()
                    if count/df_length > self.rate}
        return new_dict

    def plot(self):
        """
        Plot Pie chart.
        """
        cat_dict = self.filter()
        cat_dict = dict(sorted(cat_dict.items(),
                               key=lambda item: item[1]))
        cat = list(cat_dict.keys())
        counts = list(cat_dict.values())
        # Plot
        _, axes = plt.subplots(figsize=(8, 5),
                               subplot_kw=dict(aspect="equal"))
        title = ''.join(['Unique elements of ',
                         self.column,
                         ' (>{}%)'.format(self.rate)])
        axes.set_title(title)
        patches, _, autotexts = axes.pie(counts,
                                       autopct=lambda x: round(x, 1),
                                       startangle=90,
                                       shadow=True,
                                       wedgeprops={'edgecolor':'white',
                                                   'linewidth': 1})
        axes.legend(patches, cat, title='Categories', loc="best")
        plt.setp(autotexts, size=12, weight="bold")
        plt.show()
        return super().general_info()



# 7th tool
class CorrelationDiagram(QualityTool):
    """
    Correlation graph between two quantitative features.
    """
    def __init__(self, list_1, list_2):
        super().__init__()
        self.list1 = list_1
        self.list2 = list_2

    def keypoints_coordinates(self, list1, list2):
        """
        Return the coordinates of the four points that will form the cross,
        plus the coordinates of the intersection point.
        """
        med_1 = stat.median(list1)
        std_1 = stat.stdev(list1)
        med_2 = stat.median(list2)
        std_2 = stat.stdev(list2)
        # Points coordinates
        left_point = (med_1 - 3 * std_1, med_2)
        right_point = (med_1 + 3 * std_1, med_2)
        down_point = (med_1, med_2 - 3 * std_2)
        up_point = (med_1, med_2 + 3 * std_2)
        central_point = (med_1, med_2)
        return left_point, right_point, down_point, up_point, central_point

    def plot(self):
        """
        Plot correlation diagram.
        """
        data = pd.DataFrame()
        data['list_1'] = self.list1
        data['list_2'] = self.list2
        plot = sns.jointplot(data=data,
                             kind="reg")
        plot.plot_joint(sns.kdeplot,
                        color="r",
                        zorder=0,
                        levels=6)
