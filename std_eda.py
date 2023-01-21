# Author: B.Delorme
# Mail: delormebenoit211@gmail.com
# Creation date: 23/06/2021
# Main objective: provide a support for exploratory data analysis.

import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import statistics as stat


from collections import Counter
from matplotlib.collections import LineCollection
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from q7 import std_q7



class Sampler():
    def __init__(self, df, fraction):
        self.df = df
        self.frac = fraction

    def stratified_sampling_df(self, feature):
        categories_counter =  self.df[feature].value_counts(normalize=True)
        categories, proportions = list(categories_counter.index), list(categories_counter)
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
    def displayer(self):
        return self.EdaDisplayer(self)


    class EdaComputer():
        def __init__(self, outer):
            self.outer = outer

        def binary_dataframe(self, df):
            """Select only the binary features of the given dataframe."""
            qualitative_df = pd.DataFrame()
            for column in df.columns:
                feat_type = self.feature_type(df, column)
                if feat_type is 'binary':
                    qualitative_df[column] = df[column]
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
            """Return the feature descriptions of the given dataframe.
            - 'df' is the dataframe whose feature descriptions are wanted.
            - 'descr_df' is the dataframe that contains the feature descriptions."""
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
            """Returns the proportion of duplicates values in the given dataframe."""
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
            """Identify if there is a dominant group overwhelming the other ones."""
            counter = Counter(df[position].dropna())
            length= df.shape[0]
            signal = False
            for count in counter.values():
                if count/length > rate:
                    signal = True
            return signal

        def nan_proportion(self, df):
            """Returns the proportion of NaN values in the given dataframe."""
            nan_proportion = df.isna().sum().sum() / df.size
            nan_proportion = int(nan_proportion * 100)
            return nan_proportion

        def neat_int(self, t_int):
            """Transforms a number in a standardized integer."""
            return '{:,.0f}'.format(t_int)

        def neat_float(self, t_float):
            """Transforms a number in a standardized float."""
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
            for column in outer.self.df.columns:
                feat_type = self.feature_type(outer.self.df, column)
                if feat_type == 'quantitative':
                    quantitative_df[column] = outer.self.df[column]
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


    class EdaDisplayer():
        def __init__(self, outer):
            self.outer = outer

        def plot_feature_types(self):
            types_dict = dict(Counter(self.outer.df.dtypes))
            types = list(types_dict.keys())
            counts = list(types_dict.values())
            fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal"))
            ax.set_title('Feature types')
            patches, texts, autotexts = ax.pie(counts, autopct=lambda x: round(x, 1), startangle=90)
            ax.legend(patches, types, title='Types', loc="best")
            plt.setp(autotexts, size=12, weight="bold")
            plt.show()

        def cardinality_per_column(self):
            # Data to be plotted
            cardinalities_df = pd.DataFrame(columns=['Feature', 'Cardinality'])
            qualitative_df = self.outer.computer.qualitative_dataframe()
            features = list(qualitative_df.columns)
            cardinalities = []
            for column in features:
                cardinalities.append(qualitative_df[column].nunique())
            cardinalities_df['Feature'] = features
            cardinalities_df['Cardinality'] = cardinalities
            cardinalities_df.sort_values(by='Cardinality', ascending=True, inplace=True)
            # Plot
            plt.figure(figsize=(5, 5 + math.sqrt(5 * cardinalities_df.shape[0])))
            plt.xlim((0, 1.1 * max(cardinalities)))
            plt.title('Cardinality per feature')
            plt.barh(cardinalities_df['Feature'],
                     cardinalities_df['Cardinality'],
                     alpha=0.5, edgecolor='k')

        def dataset_infos(self):
            """Returns the main caracteristics of the given dataframe."""
            # Create the columns of the info dataframe
            info_df_columns = []
            info_df = pd.DataFrame(columns=['Rows', 'Features',
                                            'Size', 'Memory usage (bytes)',
                                            '% of NaN', '% of duplicates'],
                                   index=[df.name for df in outer.dataset])
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
            """Plot the main caracteristics of each dataframe of the given dataset.
            Enable comparison."""
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

        def plot_feature(self, column, rate=0.005, quantile_sup=1, quantile_inf=0):
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

        def plot_feature_evolution_per_datetime(self, df, column, date_column):
            featuretype = self.outer.computer.feature_type(df, column)
            if featuretype in ['quantitative', 'binary']:
                aggregated_df = df[[column, date_column]]
                aggregated_df = aggregated_df.groupby(by=date_column).sum()
                aggregated_df['Cumulated sum'] = aggregated_df[column].cumsum()
                x = list(aggregated_df.index)
                y = list(aggregated_df['Cumulated sum'])
                plt.stackplot(x, y, alpha=0.6)
            elif featuretype in ['low_cardinality', 'qualitative']:
                aggregated_df = df[[column, date_column]]
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

        def plot_feature_evolution_per_sample(self, df, column):
            featuretype = self.outer.computer.feature_type(df, column)
            if featuretype in ['quantitative', 'binary']:
                y = list(df[column].cumsum())
                plt.bar(list(range(0, len(y))), height=y, alpha=0.6)
            elif featuretype in ['low_cardinality', 'qualitative']:
                aggregated_df = pd.get_dummies(df[column], columns=[column])
                y = [aggregated_df[col].cumsum() for col in aggregated_df.columns]
                y = sorted(y, key=lambda element: max(element), reverse=True)
                plt.stackplot(list(range(0, len(df))), y, alpha=0.6, labels=aggregated_df.columns)
            else:
                return 'Feature type error'

        def plot_inflow_by_date(self, df, date_column):
            aggregated_df = df.copy()
            aggregated_df['Count by date'] = [1] * aggregated_df.shape[0]
            aggregated_df = aggregated_df.groupby(by=date_column).sum()
            aggregated_df['Cumulated sum'] = aggregated_df['Count by date'].cumsum()
            x = list(aggregated_df.index)
            plt.fill_between(x, aggregated_df['Cumulated sum'])

        def plot_nan_on_dataset(selfself, df):
            plt.figure(figsize=(10, 8))
            plt.imshow(df.isna(),
                       aspect='auto',
                       interpolation='nearest',
                       cmap='gray')
            plt.xlabel('Column number')
            plt.ylabel('Sample number')

        def plot_nan_per_sample(self, df):
            nan_proportions = []
            for i, index in enumerate(df.index):
                sample_list = df.iloc[i]
                nan_proportion = sample_list.isna().sum() / len(sample_list)
                nan_proportions.append(nan_proportion)
            # Plot
            plt.title('NaN proportion per sample')
            plt.bar(x=list(range(0, df.shape[0])), height=nan_proportions)
            #sns.displot(x=list(range(0, df.shape[0])), y=nan_proportions, kind='kde')
            #sns.distplot(a=nan_proportions, bins=list(range(0, df.shape[0])))
            #sns.histplot(a=nan_proportions, bins=list(range(0, df.shape[0])))
            #sns.kdeplot(y=nan_proportions)

        def plot_target_proportions(self, df, target_name, column, targets=[0, 1]):
            df_0 = df[df[target_name] == targets[0]]
            df_1 = df[df[target_name] == targets[1]]
            self.plot_feature(df, column)
            self.plot_feature(df_0, column)
            self.plot_feature(df_1, column)

        def qualitative_correlations_df(self, df, column_1, column_2, replace_0=False):
            """Returns a table of the correlations between categories of two qualitative
            series."""
            def correlations_dataframe(df, column_1, column_2):
                clean_df = df[[column_1, column_2]].dropna()
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
                if self.df_max(correlations_df) < 0:
                    v_max = 0
                return v_min, v_max

            correlations_df = correlations_dataframe(df, column_1, column_2)
            v_min, v_max = min_and_max(correlations_df)
            if replace_0:
                correlations_df = correlations_df.replace(0, '')
            return correlations_df.style.bar(color='lightblue', vmin=v_min, vmax=v_max)

        def qualitative_heatmap(self, df, featuretype='qualitative'):
            def cramers_v(self, serie_1, serie_2):
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

            def temp_qualitative_dataframe(df, featuretype):
                if featuretype == 'qualitative':
                    qualitative_df = self.qualitative_dataframe(df)
                elif featuretype == 'binary':
                    qualitative_df = self.binary_dataframe(df)
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

            qualitative_df = temp_qualitative_dataframe(df, featuretype)
            correlations_df = get_correlations_df(qualitative_df)
            mask = np.triu(np.ones_like(correlations_df, dtype=bool))
            factor = 5 + math.sqrt(5 * correlations_df.shape[0])
            plt.figure(figsize=(factor, factor))
            return sns.heatmap(correlations_df,
                               mask=mask, square=True, linewidths=.1, annot=False, cmap="Blues")

        def quantitative_correlations_pairplot(self, df):
            quant_df = self.outer.computer.quantitative_dataframe(df)
            sns.pairplot(quant_df,
                         height=1.5,
                         plot_kws={'s':2, 'alpha':0.2})

        def quantitative_heatmap(self, df):
            quant_df = self.outer.computer.quantitative_dataframe(df)
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
                                  cbar_kws={'shrink':0.8})
            heatmap.set_title('Correlation heatmap',
                              fontdict={'fontsize': 15}, pad=12)

        def train_test_proportion(self, train_df, test_df):
            """Plot the relative proportion of train and test set."""
            plt.title('Train / test proportion')
            plt.pie(x=[train_df.shape[0], test_df.shape[0]],
                    labels=['Train set', 'Test set'],
                    autopct=lambda x: round(x, 1),
                    startangle=90,
                    wedgeprops={'edgecolor': 'k', 'linewidth': 1})

        def violinplot(self, df, column_1, column_2):
            def median_values(df, col_quant, col_qual):
                categories = df[col_qual].unique()
                medians = [stat.median(df[df[col_qual] == category][col_quant]) for category in categories]
                return medians

            quantitative_column = [col for col in [column_1, column_2] if df[col].dtype != 'O'][0]
            return quantitative_column
            qualitative_column = [col for col in [column_1, column_2] if df[col].dtype == 'O'][0]
            medians = median_values(df, quantitative_column, qualitative_column)
            plt.plot(medians, list(range(0, len(qualitative_column)-2)),
                     color='r')
            sns.violinplot(x=quantitative_column,
                           y=qualitative_column,
                           data=df[[quantitative_column, qualitative_column]])



class FeatureEngineer():
    def categories_frequencies_dataframe(self, df, column, rate):
        categories_counter = Counter(df[column]).most_common()
        categories, occurrences = zip(*categories_counter)
        frequencies = [frequency / df.shape[0] for frequency in occurrences]
        frequencies_df = pd.DataFrame(columns=['Category', 'Frequency', 'Cumulated sum'])
        frequencies_df['Category'] = categories
        frequencies_df['Frequency'] = frequencies
        frequencies_df.sort_values(by='Frequency', ascending=False, inplace=True)
        #frequencies_df.dropna()
        frequencies_df['Cumulated sum'] = frequencies_df['Frequency'].cumsum()
        return frequencies_df

    def replace_rare_categories(self, df, column, rate, replace_by='others'):
        frequencies_df = self.categories_frequencies_dataframe(df, column, rate)
        useful_length = frequencies_df[frequencies_df['Cumulated sum'] < rate].shape[0]
        if useful_length >= 1:
            frequencies_df = frequencies_df[:useful_length + 1]
            most_frequents = list(frequencies_df['Category'])
            all_categories = list(df[column].unique())
            non_frequents = [val for val in all_categories if val not in most_frequents]
            df[column] = df[column].replace(non_frequents, replace_by)
        else:
            df.drop(column, axis=1, inplace=True)
            most_frequents = 'Not relevant. ' \
                             'Original column has been dropped from the dataframe'
        return df, most_frequents

    def replace_dates_with_differences(self, df, column):
        new_df = df.copy()
        new_df[column] = pd.to_datetime(new_df[column])
        most_recent_date = max(new_df[column])
        new_df['Oldness'] = new_df[column] - most_recent_date
        new_df['Oldness'] = [element.days for element in new_df['Oldness']]
        new_df.drop(column, axis=1, inplace=True)
        return new_df

    def log_transform(self, df, column):
        new_df = df.copy()
        column_min = min(new_df[column])
        column_max = max(new_df[column])
        # log is defined only on ]0; +infinite[
        # goal is to have column_min > 0
        if column_min <= 0:
            new_column_min = column_min + (column_max - column_min) / 1000
            new_df[column] += new_column_min
        new_df['log(' + column + ')'] = [math.log(element) for element in new_df[column]]
        new_df.drop(column, axis=1, inplace=True)
        return new_df

    def scale_transform(self, df, mode):
        new_df = df.copy()
        quant_columns = []
        for column in df.columns:
            condition_1 = (df[column].dtype != 'object')
            condition_2 = (df[column].nunique() > 4)
            if condition_1 and condition_2:
                quant_columns.append(column)
        if mode == 'std':
            scaler = StandardScaler()
        elif mode == 'minmax':
            scaler = MinMaxScaler()
        else:
            return print('Non valid mode.')
        scaled_df = scaler.fit_transform(np.array(new_df[quant_columns]))
        scaled_df = pd.DataFrame(scaled_df, columns=quant_columns)
        for column in quant_columns:
            new_df[column] = list(scaled_df[column])
        return new_df

    def split_and_scale(self, df, target):
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train = self.scale_transform(X_train, mode='minmax')
        X_test = self.scale_transform(X_test, mode='minmax')
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
        a_plot = sns.relplot(pca_df[:, 0],
                             pca_df[:, 1],
                             s=5)
        max_x = np.abs(max(pca_df[:, 0]))
        max_y = np.abs(max(pca_df[:, 1]))
        boundary = max(max_x, max_y) * 1.1
        a_plot.set(xlim=(-boundary, boundary))
        a_plot.set(ylim=(-boundary, boundary))
        return a_plot

    def scree(self, df):
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
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center')
                boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
                # Plot
                plt.xlim([-boundary, boundary])
                plt.ylim([-boundary, boundary])
                plt.plot([-100, 100], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-100, 100], color='grey', ls='--')
                plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
                plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))
                plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
                plt.show(block=False)

    def df_proper_values(self, df, n_comp):
        pca = decomposition.PCA(n_components=n_comp)
        pca.fit(df)
        return pca.singular_values_

    def draw_pca_n_dimensions(self, df, n_comp):
        """Transforme l'ensemble de données et affiche les points de données selon les deux composantes principales."""
        non_nan_df = df.dropna()
        pca = PCA(n_components=n_comp)
        pca_df = pca.fit_transform(non_nan_df)
        #
        max_x = np.abs(max(pca_df[:, 0]))
        max_y = np.abs(max(pca_df[:, 1]))
        boundary = max(max_x, max_y)*1.1
        sns.set_theme(style="darkgrid")
        a_plot = sns.relplot(pca_df[:, 0], pca_df[:, 1])
        a_plot.set(xlim=(-boundary, boundary),
                   ylim=(-boundary, boundary))
        return a_plot

    def draw_pca_circles(self, df, n_comp):
        """Affiche les deux cercles de corrélation : individus et features."""
        pca = decomposition.PCA(n_components=n_comp)
        pca.fit(df)
        pcs = pca.components_
        self.display_circles(pcs, n_comp, pca, [(0, 1)], labels=np.array(df.columns))
        plt.show()

    def pca_reduction(self, df, nb_col):
        my_scree = self.df_proper_values(df, df.shape[1])
        scree_df = pd.DataFrame({'feature': df.columns,
                                 'value': my_scree})
        # Most important n features
        best_scree_col = scree_df.sort_values(by='value', ascending=False)['feature'][:nb_col]
        return df[best_scree_col]

    def pca_reduced_df(self, df, n_comp):
        pca = PCA(n_components=n_comp)
        new_df = pd.DataFrame(pca.fit_transform(df), index=df.index)
        return new_df



class PerformancesEvaluator():
    def perf_n_pca(self, my_dataframe, pca_values_list, n_clust):
        """Returns dicts of metrics used for a list of values of principal components."""
        in_dict = {}
        ch_dict = {}
        db_dict = {}
        sil_dict = {}
        for pca_value in pca_values_list:
            my_pca_class = pca_functions.PcaViz()
            my_dataframe = my_pca_class.pca_dataframe(my_dataframe, pca_value)
            # Apply k-means
            my_kmeans_class = kmeans_functions.kmeans_op()
            my_dataframe, inertia, ch, db, sil = my_kmeans_class.kmeans_dataframe(my_dataframe, n_clust)
            in_dict[pca_value] = inertia
            ch_dict[pca_value] = ch
            db_dict[pca_value] = db
            sil_dict[pca_value] = sil
        return in_dict, ch_dict, db_dict, sil_dict

    def perf_n_clust(self, my_dataframe, n_clust_list):
        """Returns dataframes of metrics used for:
        - a list of values of principal components (columns)
        - a list of values for number of clusters (rows)"""
        n_pca_list = [650, 600, 550, 500, 450, 400, 350, 325, 300, 250, 200, 150, 100, 75, 50, 20, 10]
        in_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        ch_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        db_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        sil_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        for n_clust in n_clust_list:
            print(n_clust)
            inertia, calinksi_harsabasz, davies_bouldin, silhouette = perf_n_pca(my_dataframe, n_pca_list, n_clust)
            in_df.loc[n_clust] = list(inertia.values())
            ch_df.loc[n_clust] = list(calinksi_harabasz.values())
            db_df.loc[n_clust] = list(davies_bouldin.values())
            sil_df.loc[n_clust] = list(silhouette.values())
        return in_df, ch_df, db_df, sil_df