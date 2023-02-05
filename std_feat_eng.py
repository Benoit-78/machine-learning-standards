# -*- coding: utf-8 -*-
"""
Author: Benoît DELORME
Mail: delormebenoit211@gmail.com
Creation date: 23rd June 2021
Main objective: provide a support for feature engineering
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class FeatureEngineer():
    """
    Class designed to facilitate the Feature Engineering of a dataset.
    """
    def __init__(self, dataframe):
        """
        Dataframe to be engineered.
        """
        self.dataframe = dataframe

    def category_frequencies_df(self, column):
        """
        Return the different categories of the given qualitative column,
        with their respective frequency.
        """
        categories_counter = dict(self.dataframe[column].value_counts())
        categories = list(categories_counter.keys())
        occurrences = list(categories_counter.values())
        frequencies = [frequency / self.dataframe.shape[0]
                       for frequency in occurrences]
        frequencies_df = pd.DataFrame(columns=['Category',
                                               'Frequency',
                                               'Cumulated sum'])
        frequencies_df['Category'] = categories
        frequencies_df['Frequency'] = frequencies
        frequencies_df.sort_values(by='Frequency',
                                   ascending=False, inplace=True)
        frequencies_df['Cumulated sum'] = frequencies_df['Frequency'].cumsum()
        return frequencies_df

    def replace_rare_categories(self, column, rate=0.001):
        """
        In the given column, replaces the rare values with the strong 'others',
        to prevent umpteen rare categories to parasite future algorithm
        predictions.
        """
        count_dict = self.dataframe[column].value_counts()
        rare_values = []
        for value, count in count_dict.items():
            if count < self.dataframe.shape[0] * rate:
                rare_values.append(value)
        for rare_value in rare_values:
            self.dataframe[column] = self.dataframe[column].replace('^' + rare_value + '$',
                                                      'others', regex=True)
        return self.dataframe

    def add_date_differences(self, date_column):
        """
        Computes the difference (in days) between the date of today and all
        dates of given column.
        """
        self.dataframe[date_column] = pd.to_datetime(self.dataframe[date_column])
        most_recent_date = max(self.dataframe[date_column])
        self.dataframe['Oldness'] = self.dataframe[date_column] - most_recent_date
        self.dataframe['Oldness'] = [element.days
                                     for element in self.dataframe['Oldness']]
        return self.dataframe

    def fillna_with_trimean(self, column):
        """
        In the given quantitative column, replace NaN values with the trimean
        of the column.
        """
        quantile_25 = self.dataframe[column].quantile(0.25)
        median = self.dataframe[column].median()
        quantile_75 = self.dataframe[column].quantile(0.75)
        trimean = (quantile_25 + 2*median + quantile_75) / 4
        self.dataframe[column].fillna(trimean, inplace=True)
        return self.dataframe

    def log_transform_column(self, column):
        """
        Apply log transformation on given column.
        As log function is defined only on ]0; +infinite[, column_min value is
        set at a very small value above 0.
        """
        column_min = min(self.dataframe[column])
        if column_min < 0.:
            self.dataframe[column] += min(- column_min + 1, 1)
        elif column_min in [0., np.nan]:
            self.dataframe[column] += 1
        elif 0. < column_min < 1.:
            self.dataframe[column] += min(column_min + 1, 1)
        self.dataframe[column] = self.dataframe[column].apply(np.log)
        return self.dataframe

    def scale_columns(self, sub_df, mode):
        """
        Scale quantitative columns to help further use of gradient descent.
        """
        quant_columns = []
        for column in sub_df.columns:
            not_an_object = (self.dataframe[column].dtype != 'object')
            not_boolean = (self.dataframe[column].nunique() > 4)
            if not_an_object and not_boolean:
                quant_columns.append(column)
        if mode == 'std':
            scaler = StandardScaler()
        elif mode == 'minmax':
            scaler = MinMaxScaler()
        else:
            return print('Non valid mode.')
        scaled_df = scaler.fit_transform(np.array(self.dataframe[quant_columns]))
        scaled_df = pd.DataFrame(scaled_df, columns=quant_columns)
        for column in quant_columns:
            sub_df[column] = list(scaled_df[column])
        return sub_df

    def split_and_scale(self, target):
        """
        Split self.dataframe into 2 sets, train and test.
        Then scale train and test separately, in order to prevent data leak.
        """
        features = self.dataframe.drop(target, axis=1)
        targets = self.dataframe[target]
        [x_train, x_test,
         y_train, y_test] = train_test_split(features, targets, test_size=0.33)
        x_train = self.scale_columns(x_train, mode='minmax')
        x_test = self.scale_columns(x_test, mode='minmax')
        return x_train, x_test, y_train, y_test

    def plot_multiclass_tsne(self, column):
        """
        Plots dataset representation in t-SNE plane, with colors representing
        the different categories of the qualitative column.
        """
        feature = self.dataframe[column]
        tsne_res = TSNE(n_components=2, random_state=0)
        tsne_res.fit_transform(self.dataframe)
        tsne_res_add = np.append(tsne_res, feature, axis=1)
        n_dim = self.dataframe.shape[1] - 1
        plt.title('Groups in t-SNE plan \n{} principal components'.format(n_dim))
        n_clust = feature.unique()
        sns.scatterplot(x=tsne_res_add[:, 0],
                        y=tsne_res_add[:, 1],
                        hue=tsne_res_add[:, 2],
                        palette=sns.hls_palette(n_clust),
                        legend='full',
                        s=5)
