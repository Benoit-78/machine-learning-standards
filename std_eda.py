# -*- coding: utf-8 -*-
"""
Author: Benoît DELORME
Mail: delormebenoit211@gmail.com
Creation date: 23/06/2021
Main objective: provide a support for exploratory data analysis.
"""

import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import statistics as stat


from collections import Counter
from matplotlib.collections import LineCollection
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import std_q7
from . import std_kmeans as kmeans



class Sampler():
    def __init__(self, df, fraction):
        self.df = df
        self.frac = fraction

    def stratified_sampling_df(self, feature):
        categories_counter =  self.df[feature].value_counts(normalize=True)
        categories = list(categories_counter.index)
        new_df = pd.DataFrame()
        for category in categories:
            category_df = self.df[self.df[feature]==category]
            category_df = category_df.sample(frac=self.frac)
            new_df = pd.concat([new_df, category_df])
        return new_df

    def periodic_sampling_df(self, period):
        index_list = list(range(0, self.df.shape[0], period))
        new_df = self.df.iloc[index_list]
        return new_df



class EdaExplorator():
    def __init__(self, df):
        self.df = df

    @property
    def computer(self):
        return self.EdaComputer(self)

    @property
    def time_computer(self):
        return self.TimestampComputer(self)

    @property
    def displayer(self):
        return self.EdaDisplayer(self)

    @property
    def time_displayer(self):
        return self.TimestampDisplayer(self)


    class EdaComputer():
        def __init__(self, outer):
            self.outer = outer

        def binary_dataframe(self):
            """Select only the binary features of the given dataframe."""
            qualitative_df = pd.DataFrame()
            for column in self.outer.df.columns:
                feat_type = self.feature_type(self.outer.df, column)
                if feat_type == 'binary':
                    qualitative_df[column] = self.outer.df[column]
            return qualitative_df

        def column_from_position(self, df, position):
            for i, column in enumerate(df.columns):
                if i == position:
                    column_name = column
                    break
            return column_name

        def columns_with_potential_outliers(self, df):
            columns = []
            for column in df.columns:
                quantile_10pct = df[column].quantile(0.10)
                quantile_90pct = df[column].quantile(0.90)
                boxplot_80pct_width = quantile_90pct - quantile_10pct
                alert_on_down = (quantile_10pct - min(df[column]) > boxplot_80pct_width)
                alert_on_top = (max(df[column] - quantile_90pct) > boxplot_80pct_width)
                if alert_on_down or alert_on_top:
                    columns.append(column)
            return columns

        def dataframe_main_features(self, df, descr_df, filter_feat, ext='.csv'):
            """
            Returns the feature descriptions of the given dataframe.
            - 'df' is the dataframe whose feature descriptions are wanted.
            - 'descr_df' is the dataframe that contains the feature descriptions.
            """
            filtered_df = descr_df[descr_df[filter_feat] == df.name + ext]
            return filtered_df

        def df_list(self, my_path):
            """List of dataframes in the dataset."""
            csv_files = glob.glob(my_path + "/*.csv")
            dfs = [pd.read_csv(filename) for filename in csv_files]
            if len(dfs) == 1:
                return dfs[0]
            return dfs

        def df_max(self, df):
            """Gives the maximum value of a dataframe"""
            return max(list(df.max()))

        def df_min(self, df):
            """Gives the minimum value of a dataframe"""
            return min(list(df.min()))

        def duplicates_proportion(self, df):
            """
            Returns the proportion of duplicates values in the given dataframe.
            """
            dupl_proportion = df.duplicated().sum() / df.shape[0]
            dupl_proportion = int(dupl_proportion * 100)
            return dupl_proportion

        def feature_type(self, column):
            self.outer.df[column].dropna(inplace=True)
            unique_values = self.outer.df[column].nunique()
            if unique_values == 2:
                feat_type = 'binary'
            elif unique_values in [3, 4]:
                feat_type = 'low_cardinality'
            elif self.outer.df[column].dtype == 'object':
                feat_type = 'qualitative'
            else:
                feat_type = 'quantitative'
            return feat_type

        def is_there_a_big_group(self, df, position, rate=0.8):
            """
            Identify if there is a dominant group overwhelming the other ones.
            """
            counter = Counter(df[position].dropna())
            length= df.shape[0]
            signal = False
            for count in counter.values():
                if count/length > rate:
                    signal = True
            return signal

        def nan_proportion(self, df):
            """
            Returns the proportion of NaN values in the given dataframe.
            """
            nan_proportion = df.isna().sum().sum() / df.size
            nan_proportion = int(nan_proportion * 100)
            return nan_proportion

        def neat_int(self, t_int):
            """
            Transforms a number in a standardized integer.
            """
            return '{:,.0f}'.format(t_int)

        def neat_float(self, t_float):
            """
            Transforms a number in a standardized float.
            """
            return '{:,.2f}'.format(t_float)

        def optimize_floats(self, df):
            floats = df.select_dtypes(include=['float64']).columns.tolist()
            df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
            return df

        def optimize_ints(self, df):
            ints = df.select_dtypes(include=['int64']).columns.tolist()
            df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
            return df

        def qualitative_dataframe(self):
            """Select only the qualitative features of the given dataframe."""
            qualitative_df = pd.DataFrame()
            for column in self.outer.df.columns:
                feat_type = self.feature_type(column)
                if feat_type in ['qualitative', 'low_cardinality']:
                    qualitative_df[column] = self.outer.df[column]
            return qualitative_df

        def quantitative_dataframe(self):
            """Select only the quantitative features of the given dataframe."""
            quantitative_df = pd.DataFrame()
            for column in self.outer.df.columns:
                feat_type = self.feature_type(column)
                if feat_type == 'quantitative':
                    quantitative_df[column] = self.outer.df[column]
            return quantitative_df

        def readcsv(self, my_path, date_feature=[], dtype_dict={}, nan_values=[],
                    true_values=[], false_values=[], nrows=500):
            """Standardized csv file reading method."""
            df = pd.read_csv(my_path,
                             parse_dates=date_feature,
                             dtype=dtype_dict,
                             na_value=nan_values,
                             true_values=true_values,
                             false_values=false_values,
                             nrows=nrows)
            return df

        def fillna_with_trimean(self, column):
            quantile_25 = self.outer.df[column].quantile(0.25)
            median = self.outer.df[column].median()
            quantile_75 = self.outer.df[column].quantile(0.75)
            trimean = (quantile_25 + 2*median + quantile_75) / 4
            self.outer.df[column].fillna(trimean, inplace=True)
            return self.outer.df



    class TimestampComputer():
        def __init__(self, outer):
            self.outer = outer

        def month_occurences(self, date_column):
            month_series = self.outer.df[date_column].dt.month
            month_dict = dict(month_series.value_counts())
            month_dict = {month: count for month, count
                          in sorted(month_dict.items(), key=lambda item: item[0])}
            month_dict['January'] = month_dict.pop(1)
            month_dict['February'] = month_dict.pop(2)
            month_dict['March'] = month_dict.pop(3)
            month_dict['April'] = month_dict.pop(4)
            month_dict['May'] = month_dict.pop(5)
            month_dict['June'] = month_dict.pop(6)
            month_dict['July'] = month_dict.pop(7)
            month_dict['August'] = month_dict.pop(8)
            month_dict['September'] = month_dict.pop(9)
            month_dict['October'] = month_dict.pop(10)
            month_dict['November'] = month_dict.pop(11)
            month_dict['December'] = month_dict.pop(12)
            return month_dict
        
        def weeknumber_occurences(self, date_column):
            weeknumber_series = self.outer.df[date_column].dt.isocalendar().week
            weeknumber_dict = dict(weeknumber_series.value_counts())
            weeknumber_dict = {weeknumber: count for weeknumber, count
                          in sorted(weeknumber_dict.items(), key=lambda item: item[0])}
            return weeknumber_dict

        def hour_occurences(self, date_column):
            hour_series = self.outer.df[date_column].dt.hour
            hour_dict = dict(hour_series.value_counts())
            hour_dict = {hour: count for hour, count
                          in sorted(hour_dict.items(), key=lambda item: item[0])}
            return hour_dict

        def weekday_occurences(self, date_column):
            weekday_series = self.outer.df[date_column].dt.weekday
            weekday_dict = dict(weekday_series.value_counts())
            weekday_dict = {weekday: count for weekday, count
                          in sorted(weekday_dict.items(), key=lambda item: item[0])}
            weekday_dict['Monday'] = weekday_dict.pop(0)
            weekday_dict['Tuesday'] = weekday_dict.pop(1)
            weekday_dict['Wednesday'] = weekday_dict.pop(2)
            weekday_dict['Thursday'] = weekday_dict.pop(3)
            weekday_dict['Friday'] = weekday_dict.pop(4)
            weekday_dict['Saturday'] = weekday_dict.pop(5)
            weekday_dict['Sunday'] = weekday_dict.pop(6)
            return weekday_dict

        def monthday_occurences(self, date_column):
            day_series = self.outer.df[date_column].dt.day
            day_dict = dict(day_series.value_counts())
            day_dict = {day: count for day, count
                          in sorted(day_dict.items(), key=lambda item: item[0])}
            return day_dict

        def average_by_month(self, value_column, date_column):
            temp_df = self.outer.df[[value_column, date_column]]
            temp_df['month'] = temp_df[date_column].dt.month
            temp_df = temp_df.groupby('month').mean()
            temp_df = temp_df.reset_index()
            months = temp_df['month']
            averages = temp_df[value_column]
            month_dict = {month: average
                          for month, average in zip(months, averages)}
            month_dict['January'] = month_dict.pop(1)
            month_dict['February'] = month_dict.pop(2)
            month_dict['March'] = month_dict.pop(3)
            month_dict['April'] = month_dict.pop(4)
            month_dict['May'] = month_dict.pop(5)
            month_dict['June'] = month_dict.pop(6)
            month_dict['July'] = month_dict.pop(7)
            month_dict['August'] = month_dict.pop(8)
            month_dict['September'] = month_dict.pop(9)
            month_dict['October'] = month_dict.pop(10)
            month_dict['November'] = month_dict.pop(11)
            month_dict['December'] = month_dict.pop(12)
            return month_dict

        def average_by_weeknumber(self, value_column, date_column):
            temp_df = self.outer.df[[value_column, date_column]]
            temp_df['weeknumber'] = temp_df[date_column].dt.isocalendar().week
            temp_df = temp_df.groupby('weeknumber').mean()
            temp_df = temp_df.reset_index()
            weeknumbers = temp_df['weeknumber']
            averages = temp_df[value_column]
            weeknumber_dict = {weeknumber: average
                               for weeknumber, average in zip(weeknumbers, averages)}
            return weeknumber_dict


    class EdaDisplayer():
        def __init__(self, outer):
            self.outer = outer

        def plot_feature_types(self):
            types_dict = dict(Counter(self.outer.df.dtypes))
            types = list(types_dict.keys())
            counts = list(types_dict.values())
            fig, ax = plt.subplots(figsize=(8, 5),
                                   subplot_kw=dict(aspect="equal"))
            ax.set_title('Feature types')
            patches, texts, autotexts = ax.pie(counts, startangle=90, 
                                               autopct=lambda x: round(x, 1))
            ax.legend(patches, types, title='Types', loc="best")
            plt.setp(autotexts, size=12, weight="bold")
            plt.show()

        def cardinality_per_column(self):
            # Data to be plotted
            cardinalities_df = pd.DataFrame(columns=['Feature', 'Cardinality'])
            qualitative_df = self.outer.computer.qualitative_dataframe()
            columns = list(qualitative_df.columns)
            cardinalities = []
            for column in columns:
                cardinalities.append(qualitative_df[column].nunique())
            cardinalities_df['Column'] = columns
            cardinalities_df['Cardinality'] = cardinalities
            cardinalities_df.sort_values(by='Cardinality',
                                         ascending=True, inplace=True)
            # Plot
            plt.figure(figsize=(5, 5 + math.sqrt(5*cardinalities_df.shape[0])))
            plt.xlim((0, 1.1 * max(cardinalities)))
            plt.title('Cardinality per column')
            plt.barh(cardinalities_df['Column'],
                     cardinalities_df['Cardinality'],
                     alpha=0.5, edgecolor='k')

        def dataset_infos(self):
            """Returns the main caracteristics of the given dataframe."""
            # Create the columns of the info dataframe
            info_df = pd.DataFrame(columns=['Rows', 'Features',
                                            'Size', 'Memory usage (bytes)',
                                            '% of NaN', '% of duplicates'],
                                   index=[df.name for df in self.outer.dataset])
            # Get the data
            row_list, feature_list, size_list, nan_list, mem_list, dupl_list = [], [], [], [], [], []
            for i, df in enumerate(self.dataset):
                height = df.shape[0]
                width = df.shape[1]
                row_list.append(height)
                feature_list.append(width)
                size_list.append(df.size)
                mem_list.append(df.memory_usage(deep=True).sum())
                nan_list.append(self.nan_proportion(df))
                dupl_list.append(self.duplicates_proportion(df))
            # Constitute the dataframe
            info_df['Rows'] = row_list
            info_df['Features'] = feature_list
            info_df['Size'] = size_list
            info_df['Memory usage (bytes)'] = mem_list
            info_df['% of NaN'] = nan_list
            info_df['% of duplicates'] = dupl_list
            # Compute the average values for each feature
            average_list = []
            for feat in info_df:
                average_list.append(stat.mean(info_df[feat]))
            info_df.loc['Average'] = average_list
            return info_df.astype(int)

        def dataset_plot(self):
            """
            Plot the main caracteristics of each dataframe of the given dataset.
            Enable comparison.
            """
            info_df = self.dataset_infos()
            return info_df.style.bar(color='lightblue', align='mid')

        def nan_proportion_per_column(self):
            # Form the dataframe
            proportions_df = pd.DataFrame(columns=['Feature', 'NaN proportion', 'Color'])
            for column in self.outer.df.columns:
                if self.outer.df[column].dtype != object:
                    color = 'blue'
                else:
                    color = 'orange'
                non_nan_proportion = self.outer.df[column].notna().sum() / self.outer.df.shape[0] * 100
                proportions_df.loc[proportions_df.shape[0]] = [column, non_nan_proportion, color]
            # Filter out columns without any NaN value
            proportions_df = proportions_df[proportions_df['NaN proportion'] != 100]
            proportions_df.sort_values(by='NaN proportion', ascending=True, inplace=True)
            # Plot
            plt.figure(figsize=(5, 5 + math.sqrt(5 * proportions_df.shape[0])))
            plt.xlim((0, 105))
            plt.title('Proportion of non-NaN data \n (columns without NaN are not represented)')
            plt.barh(proportions_df['Feature'],
                     proportions_df['NaN proportion'],
                     color=proportions_df['Color'], alpha=0.5, edgecolor='k')

        def plot_feature(self, column, rate=0.001, quantile_sup=1, quantile_inf=0):
            feat_type = self.outer.computer.feature_type(column)
            if feat_type in ['binary', 'low_cardinality']:
                graph = std_q7.PieChart(self.outer.df, column, rate)
                graph.plot()
            elif feat_type == 'qualitative':
                graph = std_q7.Pareto(self.outer.df, column, rate)
                graph.plot()
            else:
                graph = std_q7.Histogram(self.outer.df, column, quantile_sup, quantile_inf)
                graph.plot()
            plt.show()

        def plot_feature_evolution_per_sample(self, column):
            featuretype = self.outer.computer.feature_type(column)
            plt.title('Feature \'{}\' evolution over samples'.format(column))
            if featuretype in ['quantitative', 'binary']:
                y = list(self.outer.df[column].cumsum())
                plt.bar(list(range(0, len(y))), height=y, alpha=0.6)
            elif featuretype in ['low_cardinality', 'qualitative']:
                aggregated_df = pd.get_dummies(self.outer.df[column], columns=[column])
                y = [aggregated_df[col].cumsum() for col in aggregated_df.columns]
                y = sorted(y, key=lambda element: max(element), reverse=True)
                plt.stackplot(list(range(0, len(self.outer.df))), y, alpha=0.6, labels=aggregated_df.columns)
            else:
                return 'Feature type error'

        def plot_inflow_by_date(self, date_column):
            aggregated_df = self.outer.df.copy()
            aggregated_df['Count by date'] = [1] * aggregated_df.shape[0]
            aggregated_df = aggregated_df.groupby(by=date_column).sum()
            aggregated_df['Cumulated sum'] = aggregated_df['Count by date'].cumsum()
            x = list(aggregated_df.index)
            plt.title('Flow of samples over time')
            plt.fill_between(x, aggregated_df['Cumulated sum'])

        def plot_feature_evolution_per_datetime(self, column, date_column):
            featuretype = self.outer.computer.feature_type(column)
            plt.title('Feature \'{}\' evolution over time'.format(column))
            if featuretype in ['quantitative', 'binary']:
                aggregated_df = self.outer.df[[column, date_column]]
                aggregated_df = aggregated_df.groupby(by=date_column).sum()
                aggregated_df['Cumulated sum'] = aggregated_df[column].cumsum()
                x = list(aggregated_df.index)
                y = list(aggregated_df['Cumulated sum'])
                plt.stackplot(x, y, alpha=0.6)
            elif featuretype in ['low_cardinality', 'qualitative']:
                aggregated_df = self.outer.df[[column, date_column]]
                aggregated_df = pd.get_dummies(aggregated_df, columns=[column])
                aggregated_df = aggregated_df.groupby(by=date_column).sum()
                for col in aggregated_df.columns:
                    aggregated_df[col] = aggregated_df[col].cumsum()
                x = list(aggregated_df.index)
                y = [list(aggregated_df[col]) for col in aggregated_df.columns]
                y = sorted(y, key=lambda element: max(element), reverse=True)
                plt.stackplot(x, y, alpha=0.6, labels=aggregated_df.columns)
            else:
                return 'Feature type error'

        def plot_nan_on_dataset(self):
            plt.figure(figsize=(10, 8))
            plt.imshow(self.outer.df.isna(),
                       aspect='auto',
                       interpolation='nearest',
                       cmap='gray')
            plt.grid(axis='x')
            plt.xlabel('Column number')
            plt.ylabel('Sample number')

        def plot_nan_per_sample(self):
            nan_proportions = []
            for i, index in enumerate(self.outer.df.index):
                sample_list = self.outer.df.iloc[i]
                nan_proportion = sample_list.isna().sum() / len(sample_list)
                nan_proportions.append(nan_proportion)
            # Plot
            plt.title('NaN proportion per sample')
            plt.ylim((0, 1.1))
            plt.bar(x=list(range(0, self.outer.df.shape[0])), height=nan_proportions)

        def plot_target_proportions(self, target_name, column, targets=[0, 1]):
            df_0 = self.df[self.df[target_name] == targets[0]]
            df_1 = self.df[self.df[target_name] == targets[1]]
            self.plot_feature(self.df, column)
            self.plot_feature(df_0, column)
            self.plot_feature(df_1, column)

        def qualitative_correlations_df(self, column_1, column_2, replace_0=False):
            """
            Returns a table of the correlations between categories of two qualitative
            series.
            """
            def correlations_dataframe(column_1, column_2):
                clean_df = self.outer.df[[column_1, column_2]].dropna()
                serie_1, serie_2 = clean_df[column_1], clean_df[column_2]
                counter_1, counter_2 = dict(Counter(serie_1)), dict(Counter(serie_2))
                keys_1, keys_2 = list(counter_1.keys()), list(counter_2.keys())
                correlations_df = pd.DataFrame(columns=keys_1, index=keys_2)
                for key_1 in keys_1:
                    t_list = []
                    for key_2 in keys_2:
                        temp_df = clean_df[clean_df[column_1]==key_1]
                        temp_df = temp_df[clean_df[column_2]==key_2]
                        t_list.append(temp_df.shape[0])
                    correlations_df[key_1] = t_list
                return correlations_df

            def min_and_max(correlations_df):
                v_min = min(list(correlations_df.min()))
                v_max = max(list(correlations_df.max()))
                if v_min > 0:
                    v_min = 0
                if self.outer.computer.df_max(correlations_df) < 0:
                    v_max = 0
                return v_min, v_max

            correlations_df = correlations_dataframe(column_1, column_2)
            v_min, v_max = min_and_max(correlations_df)
            if replace_0:
                correlations_df = correlations_df.replace(0, '')
            return correlations_df.style.bar(color='lightblue', vmin=v_min, vmax=v_max)

        def qualitative_heatmap(self, featuretype='qualitative'):
            def cramers_v(serie_1, serie_2):
                """Cramers V statistic for categorial-categorial association.
                Journal of the Korean Statistical Society 42 (2013): 323-328"""
                confusion_matrix = pd.crosstab(serie_1, serie_2)
                chi2 = ss.chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                phi2 = chi2 / n
                r, k = confusion_matrix.shape
                phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                rcorr = r - ((r - 1) ** 2) / (n - 1)
                kcorr = k - ((k - 1) ** 2) / (n - 1)
                return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

            def temp_qualitative_dataframe(featuretype):
                if featuretype == 'qualitative':
                    qualitative_df = self.outer.computer.qualitative_dataframe()
                elif featuretype == 'binary':
                    qualitative_df = self.outer.computer.binary_dataframe()
                return qualitative_df

            def get_correlations_df(qualitative_df):
                # Dataframe of coefficients
                correlations_df = pd.DataFrame(index=list(qualitative_df.columns))
                dynamic_columns_list = list(qualitative_df.columns).copy()
                for column_1 in qualitative_df.columns:
                    correlations = []
                    for column_2 in qualitative_df.columns:
                        if column_2 in dynamic_columns_list:
                            correlation = cramers_v(qualitative_df[column_1],
                                                         qualitative_df[column_2])
                            correlations.append(correlation)
                        else:
                            correlations.append(np.nan)
                    dynamic_columns_list.remove(column_1)
                    correlations_df[column_1] = correlations
                correlations_df = correlations_df.round(4)
                return correlations_df

            qualitative_df = temp_qualitative_dataframe(featuretype)
            corr_df = get_correlations_df(qualitative_df)
            labels = np.where(corr_df > 0.75, '++',
                              np.where(corr_df > 0.5, '+',
                                       np.where(corr_df < -0.5, '-',
                                                np.where(corr_df < -0.75, '--',
                                                         ''
                                                         )
                                                )
                                       )
                              )
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            factor = 5 + math.sqrt(5 * corr_df.shape[0])
            plt.figure(figsize=(factor, factor))
            heatmap = sns.heatmap(corr_df,
                                  annot=labels, mask=mask, square=True, center=0.5,
                                  linewidths=.1, cmap="Blues", fmt='',
                                  cbar_kws={'shrink':1.0})
            heatmap.set_title('Qualitative correlations heatmap',
                              fontdict={'fontsize': 15}, pad=12)

        def quantitative_correlations_pairplot(self):
            sns.pairplot(self.outer.computer.quantitative_dataframe(),
                         #height=1.5,
                         #plot_kws={'s':2, 'alpha':0.2}
                         )

        def quantitative_heatmap(self):
            quant_df = self.outer.computer.quantitative_dataframe()
            corr_df = quant_df.corr()
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            n_columns = quant_df.shape[1]
            labels = np.where(corr_df > 0.75, '++',
                              np.where(corr_df > 0.5, '+',
                                       np.where(corr_df < -0.5, '-',
                                                np.where(corr_df < -0.75, '--',
                                                         ''
                                                         )
                                                )
                                       )
                              )
            plt.figure(figsize=(n_columns * 3, n_columns * 2 / 3))
            heatmap = sns.heatmap(corr_df,
                                  annot=labels, mask=mask, square=True, center=0,
                                  linewidths=.1, cmap="vlag", fmt='',
                                  cbar_kws={'shrink':1.0})
            heatmap.set_title('Quantitative correlations heatmap',
                              fontdict={'fontsize': 15}, pad=12)

        def train_test_proportion(self, train_df, test_df):
            """Plot the relative proportion of train and test set."""
            plt.title('Train / test proportion')
            plt.pie(x=[train_df.shape[0], test_df.shape[0]],
                    labels=['Train set', 'Test set'],
                    autopct=lambda x: round(x, 1),
                    startangle=90,
                    wedgeprops={'edgecolor': 'k', 'linewidth': 1})

        def violinplot(self, column_1, column_2):
            def median_values(col_quant, col_qual):
                categories = self.outer.df[col_qual].unique()
                medians = [stat.median(self.outer.df[self.outer.df[col_qual] == category][col_quant])
                           for category in categories]
                return medians

            quantitative_column = [col for col in [column_1, column_2]
                                   if self.outer.df[col].dtype != 'O'][0]
            qualitative_column = [col for col in [column_1, column_2]
                                  if self.outer.df[col].dtype == 'O'][0]
            # medians = median_values(quantitative_column, qualitative_column)
            # plt.plot(medians, list(range(0, len(qualitative_column)-2)), color='r')
            sns.violinplot(x=quantitative_column,
                           y=qualitative_column,
                           data=self.outer.df[[quantitative_column,
                                               qualitative_column]])


    class TimestampDisplayer():
        def __init__(self, outer):
            self.outer = outer

        def plot_occurences(self, categories_dict, time_period, date_column):
            categories = list(categories_dict.keys())
            occurences = list(categories_dict.values())
            plt.title('Samples occurence over {}s, \ncolumn \'{}\''.format(time_period,
                                                                           date_column))
            plt.xticks(rotation=45)
            sns.barplot(x=categories, y=occurences,
                        edgecolor='0', color='orange', alpha=0.75)

        def plot_month_occurences(self, date_column):
            categories_dict = self.outer.time_computer.month_occurences(date_column)
            time_period = 'month'
            self.plot_occurences(categories_dict, time_period, date_column)
            
        def plot_weeknumber_occurences(self, date_column):
            categories_dict = self.outer.time_computer.weeknumber_occurences(date_column)
            time_period = 'week number'
            self.plot_occurences(categories_dict, time_period, date_column)

        def plot_hour_occurences(self, date_column):
            categories_dict = self.outer.time_computer.hour_occurences(date_column)
            time_period = 'hour'
            self.plot_occurences(categories_dict, time_period, date_column)

        def plot_weekday_occurences(self, date_column):
            categories_dict = self.outer.time_computer.weekday_occurences(date_column)
            time_period = 'week day'
            self.plot_occurences(categories_dict, time_period, date_column)

        def plot_monthday_occurences(self, date_column):
            categories_dict = self.outer.time_computer.monthday_occurences(date_column)
            time_period = 'month day'
            self.plot_occurences(categories_dict, time_period, date_column)

        def plot_averages(self, categories_dict, time_period, value_column, date_column):
            categories = list(categories_dict.keys())
            occurences = list(categories_dict.values())
            title = 'Averages of column \'{}\' over {}s.\nTime series: \'{}\''
            plt.title(title.format(value_column, time_period, date_column))
            plt.xticks(rotation=45)
            sns.barplot(x=categories, y=occurences,
                        edgecolor='0', color='blue', alpha=0.75)

        def plot_average_by_month(self, value_column, date_column):
            categories_dict = self.outer.time_computer.average_by_month(value_column,
                                                                        date_column)
            time_period = 'month'
            self.plot_averages(categories_dict, time_period,
                               value_column, date_column)

        def plot_average_by_weeknumber(self, value_column, date_column):
            categories_dict = self.outer.time_computer.average_by_weeknumber(value_column,
                                                                             date_column)
            time_period = 'weeknumber'
            self.plot_averages(categories_dict, time_period,
                               value_column, date_column)



class FeatureEngineer():
    def __init__(self, df):
        self.df = df
    
    def category_frequencies_df(self, column):
        categories_counter = dict(self.df[column].value_counts())
        categories = list(categories_counter.keys())
        occurrences = list(categories_counter.values())
        frequencies = [frequency / self.df.shape[0] for frequency in occurrences]
        frequencies_df = pd.DataFrame(columns=['Category', 'Frequency', 'Cumulated sum'])
        frequencies_df['Category'] = categories
        frequencies_df['Frequency'] = frequencies
        frequencies_df.sort_values(by='Frequency', ascending=False, inplace=True)
        #frequencies_df.dropna()
        frequencies_df['Cumulated sum'] = frequencies_df['Frequency'].cumsum()
        return frequencies_df

    def replace_rare_categories(self, column, rate=0.999, replace_by='others'):
        """
        Note: setting rate to 0.80 is equivalent to take only the most
        important categories.
        """
        frequencies_df = self.category_frequencies_df(column)
        useful_length = frequencies_df[frequencies_df['Cumulated sum'] < rate].shape[0]
        if useful_length >= 1:
            frequencies_df = frequencies_df[:useful_length + 1]
            most_frequents = list(frequencies_df['Category'])
            all_categories = list(self.df[column].unique())
            non_frequents = [val for val in all_categories
                             if val not in most_frequents]
            self.df[column] = self.df[column].replace(non_frequents, replace_by)
        else:
            self.df.drop(column, axis=1, inplace=True)
            print(most_frequents = 'Not relevant. ' \
                             'Original column has been dropped from the dataframe')
        return self.df

    def replace_rare_categories_v2(df, column, rate=0.001, replace_by='others'):
        count_dict = df[column].value_counts()
        rare_values = []
        for value, count in count_dict.items():
            if count < df.shape[0] * rate:
                rare_values.append(value)
        for rare_value in rare_values:
            df[column] = df[column].replace('^'+rare_value+'$', 'others', regex=True)
        return df

    def add_date_differences(self, date_column):
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        most_recent_date = max(self.df[date_column])
        self.df['Oldness'] = self.df[date_column] - most_recent_date
        self.df['Oldness'] = [element.days for element in self.df['Oldness']]
        return self.df

    def log_transform_column(self, column):
        # log is defined only on ]0; +infinite[
        # goal is to have column_min > 0
        column_min = min(self.df[column])
        if column_min < 0.:
            self.df[column] += min(-column_min + 1, 1)
        elif column_min == 0. or column_min == np.nan:
            self.df[column] += 1
        elif column_min > 0. and column_min < 1.:
            self.df[column] += min(column_min + 1, 1)
        self.df[column] = self.df[column].apply(np.log)
        return self.df

    def scale_column(self, df, mode):
        quant_columns = []
        for column in df.columns:
            not_an_object = (df[column].dtype != 'object')
            not_boolean = (df[column].nunique() > 4)
            if not_an_object and not_boolean:
                quant_columns.append(column)
        if mode == 'std':
            scaler = StandardScaler()
        elif mode == 'minmax':
            scaler = MinMaxScaler()
        else:
            return print('Non valid mode.')
        scaled_df = scaler.fit_transform(np.array(df[quant_columns]))
        scaled_df = pd.DataFrame(scaled_df, columns=quant_columns)
        for column in quant_columns:
            df[column] = list(scaled_df[column])
        return df

    def split_and_scale(self, target):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        X_train = self.scale_column(X_train, mode='minmax')
        X_test = self.scale_column(X_test, mode='minmax')
        return X_train, X_test, y_train, y_test



class PcaDisplayer():
    def plot_scree(self, df, n_comp):
        """Displays the pareto diagram of the proper values of a given dataframe."""
        pca = decomposition.PCA(n_components=n_comp)
        pca.fit(df)
        scree_ = pca.explained_variance_ratio_ * 100
        #
        plt.figure(figsize=(6, 8))
        plt.ylim((0, 110))
        plt.bar(np.arange(len(scree_)) + 1, scree_)
        plt.plot(np.arange(len(scree_)) + 1, scree_.cumsum(),
                 c="red", marker='o')
        plt.xlabel("Inertia axis rank")
        plt.ylabel("Inertia percentage")
        plt.title("Proper values histogram")
        plt.show(block=False)

    def plot_dataset_in_principal_plane(self, df, n_comp):
        df = df.dropna()
        pca = PCA(n_components=n_comp)
        pca_df = pca.fit_transform(df)
        sns.set_theme(style="darkgrid")
        a_plot = sns.relplot(pca_df[:, 0], pca_df[:, 1], s=5)
        max_x = np.abs(max(pca_df[:, 0]))
        max_y = np.abs(max(pca_df[:, 1]))
        boundary = max(max_x, max_y) * 1.1
        a_plot.set(xlim=(-boundary, boundary))
        a_plot.set(ylim=(-boundary, boundary))
        return a_plot

    def scree(self, df, n_comp):
        """Returns the proper values of a given dataframe."""
        pca = decomposition.PCA(n_components=n_comp)
        pca.fit(df)
        return pca.singular_values_

    def feature_circle(self, df, n_comp, pca, axis_ranks):
        """Display the correlations with arrows in the first factorial plane."""
        x_min, x_max, y_min, y_max = -1, 1, -1, 1
        for d1, d2 in axis_ranks:
            if d2 < n_comp:
                fig, ax = plt.subplots(figsize=(7, 6))
                # Features bars
                lines = [[[0, 0], [x, y]] for x, y in df[[d1, d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax,
                                                 alpha=.1, color='black'))
                # Variables names
                if labels is not None:
                    for i, (x, y) in enumerate(df[[d1, d2]].T):
                        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                            plt.text(x, y, labels[i],
                                     fontsize='14', ha='center', va='center', color="blue", alpha=0.5)
                # Plot
                circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.plot([-1, 1], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-1, 1], color='grey', ls='--')
                plt.xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
                plt.ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))
                plt.title("Cercle des corrélations (F{} et F{})".format(d1 + 1, d2 + 1))
                plt.show(block=False)

    def circles(self, X_scaled, n_comp):
        """Affiche les deux cercles de corrélation : individus et features."""
        pca = decomposition.PCA(n_components=n_comp)
        pca.fit(X_scaled)
        # pcs = pca.components_
        self.pca_feature_circle(n_comp, pca, [(0, 1)], labels=np.array(X_scaled.columns))
        plt.show()

    def plot_factorial_planes(self, X_projected, n_comp, pca, axis_ranks,
                                 labels=None, alpha=1, illustrative_var=None):
        for d1, d2 in axis_ranks:
            if d2 < n_comp:
                fig = plt.figure(figsize=(7, 6))
                # Points
                if illustrative_var is None:
                    plt.scatter(X_projected[:, d1],
                                X_projected[:, d2],
                                alpha=alpha)
                else:
                    illustrative_var = np.array(illustrative_var)
                    for value in np.unique(illustrative_var):
                        selected = np.where(illustrative_var == value)
                        plt.scatter(X_projected[selected, d1],
                                    X_projected[selected, d2],
                                    alpha=alpha, label=value)
                    plt.legend()
                # Labels
                if labels is not None:
                    for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                        plt.text(x, y, labels[i],
                                 fontsize='14', ha='center', va='center')
                boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
                # Plot
                plt.xlim([-boundary, boundary])
                plt.ylim([-boundary, boundary])
                plt.plot([-100, 100], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-100, 100], color='grey', ls='--')
                plt.xlabel('F{} ({}%)'.format(d1+1,
                                              round(100*pca.explained_variance_ratio_[d1], 1)))
                plt.ylabel('F{} ({}%)'.format(d2+1,
                                              round(100*pca.explained_variance_ratio_[d2], 1)))
                plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
                plt.show(block=False)

    def df_proper_values(self, df, n_comp):
        pca = decomposition.PCA(n_components=n_comp)
        pca.fit(df)
        return pca.mean_

    def draw_pca_circles(self, df, n_comp):
        """
        Affiche les deux cercles de corrélation : individus et features.
        """
        pca = decomposition.PCA(n_components=n_comp)
        pca.fit(df)
        pcs = pca.components_
        self.display_circles(pcs, n_comp, pca, [(0, 1)], labels=np.array(df.columns))
        plt.show()

    def first_n_features_df(self, df, new_width):
        if new_width > df.shape[1]:
            print('ERROR: New width greater than original dataframe width!')
            raise Exception()
        if new_width > df.shape[0]:
            print('WARNING: New width is smaller than original dataframe length!')
        feature_means = self.df_proper_values(df, new_width)
        scree_df = pd.DataFrame({'feature': df.columns,
                                 'mean': feature_means})
        scree_df = scree_df.sort_values(by='mean', ascending=False)
        best_scree_col = scree_df['feature'][:new_width]
        return df[best_scree_col]

    def pca_reduced_df(self, df, n_comp):
        pca = PCA(n_components=n_comp)
        new_df = pd.DataFrame(pca.fit_transform(df), index=df.index)
        return new_df



class PerformancesEvaluator():
    def perf_n_pca(self, df, pca_values_list, n_clust):
        """
        Returns dicts of metrics used for a list of values of principal
        components.
        """
        in_dict = {}
        ch_dict = {}
        db_dict = {}
        sil_dict = {}
        for pca_value in pca_values_list:
            print('Number of pca components:', pca_value)
            PcaClass = PcaDisplayer()
            pca_df = PcaClass.first_n_features_df(df, pca_value)
            # Apply k-means
            Identifier = kmeans.KmeansIdentifier(pca_df)
            model = Identifier.get_model_and_add_labels(n_clust)
            [inertia, ch, db, sil] = Identifier.get_kmeans_metrics(model)
            in_dict[pca_value] = inertia
            ch_dict[pca_value] = ch
            db_dict[pca_value] = db
            sil_dict[pca_value] = sil
        return [in_dict, ch_dict, db_dict, sil_dict]

    def perf_n_clust(self, df, n_clust_list):
        """
        Returns dataframes of metrics used for:
        - a list of values of principal components (columns)
        - a list of values for number of clusters (rows)
        """
        df_width = df.shape[1]
        n_pca_list = [df_width,
                      int(df_width*0.5),
                      int(df_width*0.2),
                      int(df_width*0.1),
                      int(df_width*0.05),
                      int(df_width*0.02),
                      int(df_width*0.01)]
        in_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        ch_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        db_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        sil_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        for n_clust in n_clust_list:
            print('Number of clusters:', n_clust)
            [inertia,
             calinski_harabasz,
             davies_bouldin,
             silhouette] = self.perf_n_pca(df, n_pca_list, n_clust)
            in_df.loc[n_clust] = list(inertia.values())
            ch_df.loc[n_clust] = list(calinski_harabasz.values())
            db_df.loc[n_clust] = list(davies_bouldin.values())
            sil_df.loc[n_clust] = list(silhouette.values())
        return in_df, ch_df, db_df, sil_df


def plot_multiclass_tsne(X, y):
    tsne_res = TSNE(n_components=2, random_state=0).fit_transform(X)
    # labels = np.expand_dims(y, axis=1)
    tsne_res_add = np.append(tsne_res, y, axis=1)
    n_dim = X.shape[1] - 1
    plt.title('Groups in t-SNE plan \n{} principal components'.format(n_dim))
    n_clust = len(list(y[y.columns[0]].unique()))
    sns.scatterplot(x=tsne_res_add[:, 0],
                    y=tsne_res_add[:, 1],
                    hue=tsne_res_add[:, 2],
                    palette=sns.hls_palette(n_clust),
                    legend='full',
                    s=5)


def show_lime_in_notebook(y_train, X_test):
    from lime.lime_text import LimeTextExplainer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    class_names = list(y_train.unique())
    explainer = LimeTextExplainer(class_names=class_names)
    #
    vectorizer = TfidfVectorizer(lowercase=False)
    nb = MultinomialNB(alpha=.01)
    c = make_pipeline(vectorizer, nb)
    #
    idx = 1340
    #
    exp = explainer.explain_instance(X_test.data[idx],
                                     c.predict_proba,
                                     num_features=6,
                                     top_labels=2)