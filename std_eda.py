# -*- coding: utf-8 -*-
"""
Author: Benoît DELORME
Mail: delormebenoit211@gmail.com
Creation date: 23rd June 2021
Main objective: provide a support for exploratory data analysis.
"""

import math
import statistics as stat

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from q7 import std_q7 as q7



class Sampler():
    """
    Provide sampling strategies.
    """
    def __init__(self, dataframe, fraction):
        """
        df :
            pd.DataFrame
        fraction : float in [0; 1]
            Fraction of the dataframe taken.
        """
        self.dataframe = dataframe
        self.frac = fraction

    def stratified_sampling_df(self, feature):
        """
        Parameters
        ----------
        feature : string

        Returns
        -------
        new_df : pd.DataFrame
            Stratified dataframe.
        """
        cat_counter = self.dataframe[feature].value_counts(normalize=True)
        categories = list(cat_counter.index)
        new_df = pd.DataFrame()
        for category in categories:
            category_df = self.dataframe[self.dataframe[feature]==category]
            category_df = category_df.sample(frac=self.frac)
            new_df = pd.concat([new_df, category_df])
        return new_df

    def periodic_sampling_df(self, period):
        """
        Parameters
        ----------
        period : int
            Interval between each sample.

        Returns
        -------
        new_df :pd.DataFrame
            Periodic sampled dataframe.
        """
        index_list = list(range(0, self.dataframe.shape[0], period))
        new_dataframe = self.dataframe.iloc[index_list]
        return new_dataframe



class EdaExplorator():
    """
    Class designed to facilitate the Exploratory Data Analysis of a dataset.
    """
    def __init__(self, df, dataset=None):
        """
        df : pd.Dataframe
            Dataframe on which EDA is done.
        dataset : dict
            Set of dataframes on which EDA is done.
        """
        self.df = df
        self.dataset = dataset

    @property
    def computer(self):
        """
        Property aimed at simplifying user's input.
        """
        return self.EdaComputer(self)

    @property
    def time_computer(self):
        """
        Property aimed at simplifying user's input.
        """
        return self.TimestampComputer(self)

    @property
    def displayer(self):
        """
        Property aimed at simplifying user's input.
        """
        return self.EdaDisplayer(self)

    @property
    def time_displayer(self):
        """
        Property aimed at simplifying user's input.
        """
        return self.TimestampDisplayer(self)


    class EdaComputer():
        """
        Provide with computation methods, whose results are used by
        EdaDisplayer methods for visualisation purposes.
        """
        def __init__(self, outer):
            """
            Retrieve mother class's arguments.
            """
            self.outer = outer

        def binary_dataframe(self):
            """
            Select only the binary features of the given dataframe.
            """
            qualitative_df = pd.DataFrame()
            for column in self.outer.df.columns:
                feat_type = self.feature_type(column)
                if feat_type == 'binary':
                    qualitative_df[column] = self.outer.df[column]
            return qualitative_df

        def columns_with_potential_outliers(self):
            """
            Identify columns that are susceptible to contain outliers.
            """
            columns = []
            for column in self.outer.df.columns:
                quant_10 = self.outer.df[column].quantile(0.10)
                quant_90 = self.outer.df[column].quantile(0.90)
                width_80 = quant_90 - quant_10
                down_alert = (quant_10 - min(self.outer.df[column]) > width_80)
                top_alert = (max(self.outer.df[column] - quant_90) > width_80)
                if down_alert or top_alert:
                    columns.append(column)
            return columns

        def dataframe_main_features(self, descr_df, col, ext='.csv'):
            """
            Return the feature descriptions of the given dataframe.

            descr_df : dataframe containing the feature descriptions.
            """
            filtered_df = descr_df[descr_df[col] == self.outer.df.name + ext]
            return filtered_df

        def df_max(self):
            """
            Gives the maximum value of a dataframe
            """
            return max(list(self.outer.df.max()))

        def df_min(self):
            """
            Gives the minimum value of a dataframe
            """
            return min(list(self.outer.df.min()))

        def duplicates_proportion(self):
            """
            Returns the proportion of duplicates values in the given dataframe.
            """
            duplicates_total = self.outer.df.duplicated().sum()
            df_length = self.outer.df.shape[0]
            dupl_proportion = int(duplicates_total / df_length * 100)
            return dupl_proportion

        def feature_type(self, column):
            """
            Return the type of the given column.

            Can be either binary, quantitative, qualitative, or qualitative
            of low cardinality (low_cardinality).
            """
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

        def is_there_a_big_group(self, position, rate=0.8):
            """
            Identify if there is a dominant group overwhelming the other ones.
            """
            counter = Counter(self.outer.df[position].dropna())
            length= self.outer.df.shape[0]
            signal = False
            for count in counter.values():
                if count/length > rate:
                    signal = True
            return signal

        def nan_proportion(self):
            """
            Returns the proportion of NaN values in the given dataframe.
            """
            total_nan = self.outer.df.isna().sum().sum()
            df_length = self.outer.df.size
            nan_proportion = int(total_nan / df_length * 100)
            return nan_proportion

        def optimize_floats(self):
            """
            Reduce the memory size taken by float columns.
            """
            float_df = self.outer.df.select_dtypes(include=['float64'])
            cols = float_df.columns.tolist()
            self.outer.df[cols] = self.outer.df[cols].apply(pd.to_numeric,
                                                              downcast='float')
            return self.outer.df

        def optimize_ints(self):
            """
            Reduce the memory size taken by integer columns.
            """
            int_df = self.outer.df.select_dtypes(include=['int64'])
            cols = int_df.columns.tolist()
            self.outer.df[cols] = self.outer.df[cols].apply(pd.to_numeric,
                                                            downcast='integer')
            return self.outer.df

        def qualitative_dataframe(self):
            """
            Select only the qualitative features of the given dataframe.
            """
            qualitative_df = pd.DataFrame()
            for column in self.outer.df.columns:
                feat_type = self.feature_type(column)
                if feat_type in ['qualitative', 'low_cardinality']:
                    qualitative_df[column] = self.outer.df[column]
            return qualitative_df

        def quantitative_dataframe(self):
            """
            Select only the quantitative features of the given dataframe.
            """
            quantitative_df = pd.DataFrame()
            for column in self.outer.df.columns:
                feat_type = self.feature_type(column)
                if feat_type == 'quantitative':
                    quantitative_df[column] = self.outer.df[column]
            return quantitative_df



    class TimestampComputer():
        """
        Provide with time computation methods, whose results are used by
        TimeStampDisplayer methods for visualisation purposes.
        """
        def __init__(self, outer):
            """
            Retrieve mother class's arguments.
            """
            self.outer = outer

        def month_occurences(self, date_column):
            """
            Return occurences of each of the 12 months in the given date
            column.
            There is no distinction between different years.
            """
            month_series = self.outer.df[date_column].dt.month
            month_dict = dict(month_series.value_counts())
            month_dict = dict(sorted(month_dict.items(),
                                     key=lambda item: item[1]))
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

        def weeknb_occurences(self, date_column):
            """
            Return count per week number (1-52).
            """
            weeknb_series = self.outer.df[date_column].dt.isocalendar().week
            weeknb_dict = dict(weeknb_series.value_counts())
            weeknb_dict = dict(sorted(weeknb_dict.items(),
                                      key=lambda item: item[1]))
            return weeknb_dict

        def hour_occurences(self, date_column):
            """
            Return count per hour (0-23).
            """
            hour_series = self.outer.df[date_column].dt.hour
            hour_dict = dict(hour_series.value_counts())
            hour_dict = dict(sorted(hour_dict.items(),
                                    key=lambda item: item[1]))
            return hour_dict

        def weekday_occurences(self, date_column):
            """
            Return count per weekday (0-6).
            """
            weekday_series = self.outer.df[date_column].dt.weekday
            weekday_dict = dict(weekday_series.value_counts())
            weekday_dict = dict(sorted(weekday_dict.items(),
                                       key=lambda item: item[1]))
            weekday_dict['Monday'] = weekday_dict.pop(0)
            weekday_dict['Tuesday'] = weekday_dict.pop(1)
            weekday_dict['Wednesday'] = weekday_dict.pop(2)
            weekday_dict['Thursday'] = weekday_dict.pop(3)
            weekday_dict['Friday'] = weekday_dict.pop(4)
            weekday_dict['Saturday'] = weekday_dict.pop(5)
            weekday_dict['Sunday'] = weekday_dict.pop(6)
            return weekday_dict

        def monthday_occurences(self, date_column):
            """
            Return count per monthday (1-31).
            """
            day_series = self.outer.df[date_column].dt.day
            day_dict = dict(day_series.value_counts())
            day_dict = dict(sorted(day_dict.items(),
                                   key=lambda item: item[1]))
            return day_dict

        def average_by_month(self, value_column, date_column):
            """
            Return average of given value_column per month. The month taken
            from date column.
            """
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

        def average_by_week(self, value_column, date_column):
            """
            Return average of given value_column per week. The week is taken
            from date column.
            """
            temp_df = self.outer.df[[value_column, date_column]]
            temp_df['weeknumber'] = temp_df[date_column].dt.isocalendar().week
            temp_df = temp_df.groupby('weeknumber').mean()
            temp_df = temp_df.reset_index()
            weeknumbers = temp_df['weeknumber']
            averages = temp_df[value_column]
            weeknumber_dict = {weeknumber: average
                               for weeknumber, average in zip(weeknumbers,
                                                              averages)}
            return weeknumber_dict



    class EdaDisplayer():
        """
        Provide visualisation tools for Exploratory Data Analysis.
        """
        def __init__(self, outer):
            """
            Retrieve mother class's arguments.
            """
            self.outer = outer

        def dataset_plot(self):
            """
            Plot main caracteristics of each dataframe of the given dataset,
            enabling comparison between dataframes.
            """
            info_df = self.dataset_infos()
            return info_df.style.bar(color='lightblue', align='mid')

        def plot_nan_on_dataframe(self):
            """
            Plot an overview of dataframe, with data as black, and NaN as
            white.
            """
            plt.figure(figsize=(10, 8))
            plt.imshow(self.outer.df.isna(),
                       aspect='auto',
                       interpolation='nearest',
                       cmap='gray')
            plt.grid(axis='x')
            plt.tick_params(axis="x", bottom=True, top=True,
                            labelbottom=True, labeltop=True)
            plt.xlabel('Column number')
            plt.ylabel('Sample number')

        def plot_data_per_sample(self, chunk=20):
            """
            Plot the proportion of non-NaN data per sample.
            """
            nan_sum = list(self.outer.df.notna().sum(axis=1))
            width = self.outer.df.shape[1]
            nan_proportions = [element / width for element in nan_sum]
            nan_s = pd.Series(nan_proportions)
            nan_s = nan_s.groupby(np.arange(len(nan_s)) // chunk).mean()
            plt.figure(figsize=(15, 5))
            msg = 'Data proportion per sample (non NaN)'
            msg += '\nAverages every {} samples.'
            plt.title(msg.format(chunk))
            plt.grid(axis='x')
            plt.grid(axis='y')
            plt.ylim((0, 1.1))
            x_axis = [i*chunk for i, _ in enumerate(nan_s)]
            plt.plot(x_axis, nan_s, linewidth=0,
                     marker='o', markersize=1)

        def plot_memory_usage_per_sample(self, chunk=20):
            memories = []
            for i in range(self.outer.df.shape[0]):
                memory = self.outer.df.iloc[i].memory_usage(index=True,
                                                            deep=True)
                # pandas returns bytes by defaults.
                memory = memory / 1024 ** 2
                memories.append(memory)
            # Average
            memories = pd.Series(memories)
            memories = memories.groupby(np.arange(memories.shape[0]) // chunk).mean()
            # Plot
            plt.figure(figsize=(15, 5))
            plt.title('Memory per sample\nAverages on {} samples.'.format(chunk))
            plt.grid(axis='x')
            plt.grid(axis='y')
            plt.ylim((0, 1.1 * max(memories)))
            x_axis = [i * chunk for i, _ in enumerate(memories)]
            plt.plot(x_axis, memories, linewidth=0,
                     marker='o', markersize=1)

        def plot_data_per_column(self):
            """
            Plot the proportion of non-NaN data per column.
            Quantitative columns are orange.
            Qualitatitve columns are blue.
            """
            # Form the dataframe
            proportions_df = pd.DataFrame(columns=['Feature',
                                                   'NaN proportion',
                                                   'Color'])
            for column in self.outer.df.columns:
                if self.outer.df[column].dtype != object:
                    color = 'blue'
                else:
                    color = 'orange'
                nan_total = self.outer.df[column].notna().sum()
                df_shape = self.outer.df.shape[0]
                non_nan_prop = nan_total / df_shape * 100
                proportions_df.loc[proportions_df.shape[0]] = [column,
                                                               non_nan_prop,
                                                               color]
            # Filter out columns without any NaN value
            proportions_df = proportions_df[proportions_df['NaN proportion'] != 100]
            proportions_df.sort_values(by='NaN proportion',
                                       ascending=True, inplace=True)
            # Plot
            plt.figure(figsize=(5, 5 + math.sqrt(5 * proportions_df.shape[0])))
            plt.xlim((0, 1.05 * 100))
            plt.tick_params(axis="x",
                            bottom=True, top=True,
                            labelbottom=True, labeltop=True)
            plt.grid(axis='x')
            plt.title('Proportion of non-NaN data \n (complete columns not represented)')
            plt.barh(proportions_df['Feature'],
                     proportions_df['NaN proportion'],
                     color=proportions_df['Color'], alpha=0.5, edgecolor='k')

        def plot_memory_usage_per_column(self):
            """
            Plot the memory usage per column, and indicates the total memory
            taken by the whol dataframe.
            """
            memory_df = pd.DataFrame(columns=['Feature', 'Memory', 'Color'])
            for column in self.outer.df.columns:
                if self.outer.df[column].dtype != object:
                    color = 'blue'
                else:
                    color = 'orange'
                memory = self.outer.df[column].memory_usage(index=True,
                                                            deep=True)
                # pandas returns bytes by defaults.
                memory = memory / 1024**2
                memory = round(memory, 2)
                row = pd.DataFrame([[column, memory, color]],
                                   columns=['Feature', 'Memory', 'Color'])
                memory_df = pd.concat([memory_df, row])
            memory_df = memory_df.sort_values(by='Memory', ascending=True)
            total_memory = self.outer.df.memory_usage(index=True,
                                                      deep=True).sum()
            total_memory = total_memory / 1024**2
            total_memory = round(total_memory , 1)
            # Plot
            plt.figure(figsize=(5, 5 + math.sqrt(5 * memory_df.shape[0])))
            max_memory = max(memory_df['Memory'])
            plt.xlim((0, 1.05 * max_memory))
            plt.tick_params(axis="x", bottom=True, top=True,
                            labelbottom=True, labeltop=True)
            plt.grid(axis='x')
            msg = 'Memory usage per column (MB)\nTotal: {} MB'
            plt.title(msg.format(total_memory))
            plt.barh(memory_df['Feature'],
                     memory_df['Memory'],
                     color=memory_df['Color'],
                     alpha=0.5, edgecolor='k')

        def plot_cardinality_per_column(self):
            """
            Plot the cardinality for each qualitative column.
            """
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
            plt.xlim((0, 1.05 * max(cardinalities)))
            plt.tick_params(axis="x",
                            bottom=True, top=True,
                            labelbottom=True, labeltop=True)
            plt.grid(axis='x')
            plt.title('Cardinality per column')
            plt.barh(cardinalities_df['Column'],
                     cardinalities_df['Cardinality'],
                     alpha=0.5, edgecolor='k')

        def plot_feature_types(self):
            """
            Plot a pie chart representing the proportion of each type (integer,
            float, object, ...) of dataframe columns.
            """
            types_dict = dict(Counter(self.outer.df.dtypes))
            types = list(types_dict.keys())
            counts = list(types_dict.values())
            _, axis = plt.subplots(figsize=(8, 5),
                                   subplot_kw=dict(aspect="equal"))
            axis.set_title('Feature types')
            patches, _, autotexts = axis.pie(counts, startangle=90,
                                               autopct=lambda x: round(x, 1))
            axis.legend(patches, types, title='Types', loc="best")
            plt.setp(autotexts, size=12, weight="bold")
            plt.show()

        def plot_feature(self, column, rate=0.001, quant_sup=1, quant_inf=0):
            """
            Return a visual representation for the given column.
            The representation is specific to the column type, which can be:
                - binary,
                - quantitative,
                - qualitative,
                - or qualitative of low cardinality.
            """
            feat_type = self.outer.computer.feature_type(column)
            if feat_type in ['binary', 'low_cardinality']:
                graph = q7.PieChart(self.outer.df, column, rate)
                graph.plot()
            elif feat_type == 'qualitative':
                graph = q7.Pareto(self.outer.df, column, rate)
                graph.plot()
            else:
                graph = q7.Histogram(self.outer.df, column,
                                     quant_sup, quant_inf)
                graph.plot()
            plt.show()

        def plot_feature_evolution_per_sample(self, column):
            """
            For the given column, plot the evolution of values along all
            samples.
            """
            featuretype = self.outer.computer.feature_type(column)
            plt.title('Feature \'{}\' evolution over samples'.format(column))
            if featuretype in ['quantitative', 'binary']:
                values = list(self.outer.df[column].cumsum())
                plt.bar(list(range(0, len(values))), height=values, alpha=0.6)
                result = None
            elif featuretype in ['low_cardinality', 'qualitative']:
                aggreg_df = pd.get_dummies(self.outer.df[column],
                                           columns=[column])
                values = [aggreg_df[col].cumsum() for col in aggreg_df.columns]
                values = sorted(values, key=lambda element: max(element),
                                reverse=True)
                plt.stackplot(list(range(0, self.outer.df.shape[0])),
                              values,
                              alpha=0.6,
                              labels=aggreg_df.columns)
                result = None
            else:
                result = 'Feature type error'
            return result

        def plot_feature_evolution_per_datetime(self, column, date_column):
            """
            For the given column, plot the evoluation of values along the given
            date column.
            """
            featuretype = self.outer.computer.feature_type(column)
            plt.title('Feature \'{}\' evolution over time'.format(column))
            if featuretype in ['quantitative', 'binary']:
                aggregated_df = self.outer.df[[column, date_column]]
                aggregated_df = aggregated_df.groupby(by=date_column).sum()
                aggregated_df['Cumulated sum'] = aggregated_df[column].cumsum()
                dates = list(aggregated_df.index)
                values = list(aggregated_df['Cumulated sum'])
                plt.stackplot(dates, values, alpha=0.6)
                result = None
            elif featuretype in ['low_cardinality', 'qualitative']:
                aggregated_df = self.outer.df[[column, date_column]]
                aggregated_df = pd.get_dummies(aggregated_df, columns=[column])
                aggregated_df = aggregated_df.groupby(by=date_column).sum()
                for col in aggregated_df.columns:
                    aggregated_df[col] = aggregated_df[col].cumsum()
                dates = list(aggregated_df.index)
                values = [list(aggregated_df[col]) for col in aggregated_df.columns]
                values = sorted(values, key=lambda element: max(element), reverse=True)
                plt.stackplot(dates, values, alpha=0.6, labels=aggregated_df.columns)
                result = None
            else:
                result = 'Feature type error'
            return result

        def plot_inflow_by_date(self, date_column):
            """
            Plot the amount of samples per day. The day is extracted from the
            given date_column.
            """
            aggreg_df = self.outer.df.copy()
            aggreg_df['Count by date'] = [1] * aggreg_df.shape[0]
            aggreg_df = aggreg_df.groupby(by=date_column).sum()
            aggreg_df['Cumulated sum'] = aggreg_df['Count by date'].cumsum()
            dates = list(aggreg_df.index)
            plt.title('Flow of samples over time')
            plt.fill_between(dates, aggreg_df['Cumulated sum'])

        def qualitative_heatmap(self, featuretype='qualitative'):
            """
            Return a heatmap of correlations between qualitative columns.
            """
            def cramers_v(serie_1, serie_2):
                """Cramers V statistic for categorial-categorial association.
                Journal of the Korean Statistical Society 42 (2013): 323-328"""
                confusion_matrix = pd.crosstab(serie_1, serie_2)
                chi2 = ss.chi2_contingency(confusion_matrix)[0]
                sumsum = confusion_matrix.sum().sum()
                phi2 = chi2 / sumsum
                dim_1, dim_2 = confusion_matrix.shape
                phi2corr = max(0,
                               phi2 - ((dim_2 - 1)*(dim_1 - 1)) / (sumsum - 1))
                rcorr = dim_1 - ((dim_1 - 1) ** 2) / (sumsum - 1)
                kcorr = dim_2 - ((dim_2 - 1) ** 2) / (sumsum - 1)
                return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

            def temp_qualitative_dataframe(featuretype):
                """
                Return a fraction of self.dataframe, with qualitative columns
                only.
                """
                if featuretype == 'qualitative':
                    qualitative_df = self.outer.computer.qualitative_dataframe()
                elif featuretype == 'binary':
                    qualitative_df = self.outer.computer.binary_dataframe()
                return qualitative_df

            def get_correlations_df(qualitative_df):
                """
                Return a dataframe containing the correlation between
                qualitative features of self.dataframe.
                """
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
                                  annot=labels, mask=mask, square=True,
                                  center=0.5, linewidths=.1, cmap="Blues",
                                  fmt='', cbar_kws={'shrink':1.0})
            heatmap.set_title('Qualitative correlations heatmap',
                              fontdict={'fontsize': 15}, pad=12)

        def quantitative_heatmap(self):
            """
            Return a heatmap of correlations between quantitative columns.
            """
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
                                  annot=labels, mask=mask, square=True,
                                  center=0, linewidths=.1, cmap="vlag", fmt='',
                                  cbar_kws={'shrink':1.0})
            heatmap.set_title('Quantitative correlations heatmap',
                              fontdict={'fontsize': 15}, pad=12)

        def quantitative_correlations_pairplot(self):
            """
            Pairplot of correlations between quantitative columns.
            """
            sns.pairplot(self.outer.computer.quantitative_dataframe(),
                         #height=1.5,
                         #plot_kws={'s':2, 'alpha':0.2}
                         )

        def qualitative_correlations_df(self, column_1, column_2, replace_0=False):
            """
            Returns a table of the correlations between categories of two
            qualitative series.
            """
            def correlations_dataframe(column_1, column_2):
                clean_df = self.outer.df[[column_1, column_2]].dropna()
                serie_1, serie_2 = clean_df[column_1], clean_df[column_2]
                counter_1 = dict(Counter(serie_1))
                counter_2 = dict(Counter(serie_2))
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
            return correlations_df.style.bar(color='lightblue',
                                             vmin=v_min, vmax=v_max)

        def violinplot(self, column_1, column_2):
            """
            1) On the two columns, one is qualitative, and the other one
            quantitative
            2) Return a violinplot, representing the correlations between
            categories of the qualitative column, and the quantitative values.
            """
            quantitative_column = [col for col in [column_1, column_2]
                                   if self.outer.df[col].dtype != 'O'][0]
            qualitative_column = [col for col in [column_1, column_2]
                                  if self.outer.df[col].dtype == 'O'][0]
            sns.violinplot(x=quantitative_column,
                           y=qualitative_column,
                           data=self.outer.df[[quantitative_column,
                                               qualitative_column]])

        def dataset_infos(self):
            """
            Returns the main caracteristics of the given dataframe.
            """
            # Create the columns of the info dataframe
            info_df = pd.DataFrame(columns=['Rows', 'Features',
                                            'Size', 'Memory usage (bytes)',
                                            '% of NaN', '% of duplicates'],
                                   index=[df.name
                                          for df in self.outer.dataset])
            # Get the data
            row_list = []
            feature_list = []
            size_list = []
            nan_list = []
            mem_list = []
            dupl_list = []
            for dataframe in self.outer.dataset:
                height = dataframe.shape[0]
                width = dataframe.shape[1]
                row_list.append(height)
                feature_list.append(width)
                size_list.append(dataframe.size)
                mem_list.append(dataframe.memory_usage(deep=True).sum())
                nan_list.append(self.outer.computer.nan_proportion(dataframe))
                dupl_list.append(self.outer.computer.duplicates_proportion(
                                                                    dataframe))
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



    class TimestampDisplayer():
        """
        Provide visualisation tools for Exploratory Data Analysis.
        """
        def __init__(self, outer):
            """
            Retrieve mother class's arguments.
            """
            self.outer = outer

        def plot_occurences(self, cat_dict, time_period, date_column):
            """
            Back-end visualisation tool for time data.
            To be used by methods of class TimeStampDisplayer.
            """
            categories = list(cat_dict.keys())
            occurences = list(cat_dict.values())
            msg = 'Samples occurence over {}s, \ncolumn \'{}\''
            plt.title(msg.format(time_period, date_column))

        def plot_date_formats(self, date_columns):
            """
            Display the different formats of date in the dataframe.
            """
            spec_char = ['/', ':']
            french_months = {'January': 'Janvier',
                             'February': 'Février',
                             'March': 'Mars',
                             'April': 'Avril',
                             'May': 'Mai',
                             'June': 'Juin',
                             'July': 'Juillet',
                             'August': 'Août',
                             'September': 'Septembre',
                             'October': 'Octobre',
                             'November': 'Novembre',
                             'December': 'Décembre'}
            temp_df = self.df.sample(frac=0.1)
            for column in date_columns:
                print(column)
            plt.xticks(rotation=45)
            sns.barplot(x=categories, y=occurences,
                        edgecolor='0', color='orange', alpha=0.75)

        def plot_month_occurences(self, date_column):
            """
            Tool for visualisation of months repartition.
            """
            cat_dict = self.outer.time_computer.month_occurences(date_column)
            time_period = 'month'
            self.plot_occurences(cat_dict, time_period, date_column)

        def plot_weeknb_occurences(self, date_column):
            """
            Tool for visualisation of weeks repartition.
            """
            cat_dict = self.outer.time_computer.weeknb_occurences(date_column)
            time_period = 'week number'
            self.plot_occurences(cat_dict, time_period, date_column)

        def plot_hour_occurences(self, date_column):
            """
            Tool for visualisation of hours repartition.
            """
            cat_dict = self.outer.time_computer.hour_occurences(date_column)
            time_period = 'hour'
            self.plot_occurences(cat_dict, time_period, date_column)

        def plot_weekday_occurences(self, date_column):
            """
            Tool for visualisation of weekdays repartition.
            """
            cat_dict = self.outer.time_computer.weekday_occurences(date_column)
            time_period = 'week day'
            self.plot_occurences(cat_dict, time_period, date_column)

        def plot_monthday_occurences(self, date_column):
            """
            Tool for visualisation of monthdays repartition.
            """
            cat_dict = self.outer.time_computer.monthday_occurences(date_column)
            time_period = 'month day'
            self.plot_occurences(cat_dict, time_period, date_column)

        def plot_averages(self, cat_dict, time_period, value_col, date_col):
            """
            Back-end visualisation tool for time data.
            To be used by methods of class TimeStampDisplayer.
            """
            categories = list(cat_dict.keys())
            occurences = list(cat_dict.values())
            title = 'Averages of column \'{}\' over {}s.\nTime series: \'{}\''
            plt.title(title.format(value_col, time_period, date_col))
            plt.xticks(rotation=45)
            sns.barplot(x=categories, y=occurences,
                        edgecolor='0', color='blue', alpha=0.75)

        def plot_average_by_month(self, value_column, date_column):
            """
            Tool for visualisation of average value of given column for each
            month.
            """
            cat_dict = self.outer.time_computer.average_by_month(value_column,
                                                                 date_column)
            time_period = 'month'
            self.plot_averages(cat_dict, time_period,
                               value_column, date_column)

        def plot_average_by_weeknumber(self, value_column, date_column):
            """
            Tool for visualisation of average value of given column for each
            week.
            """
            cat_dict = self.outer.time_computer.average_by_week(value_column,
                                                                date_column)
            time_period = 'weeknumber'
            self.plot_averages(cat_dict, time_period,
                               value_column, date_column)
