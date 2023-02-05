# -*- coding: utf-8 -*-
"""
Author: Benoît DELORME
Mail: delormebenoit211@gmail.com
Creation date: 23rd June 2021
Main objective: provide a support for use PCA reduction.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.collections import LineCollection
from sklearn import decomposition
from sklearn.decomposition import PCA

import std_kmeans

class PcaDisplayer():
    """
    Class designed to facilitate the Feature Engineering of a dataset.
    """
    def __init__(self, dataframe, n_comp):
        """
        Dataframe on which the PCA is performed, with the dimension to which it
        is reduced.
        """
        self.dataframe = dataframe
        self.n_comp = n_comp

    def plot_scree(self):
        """
        Displays the pareto diagram of the proper values of a given dataframe.
        """
        pca = decomposition.PCA(n_components=self.n_comp)
        pca.fit(self.dataframe)
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

    def plot_dataset_in_principal_plane(self):
        """
        Plot dataset in principal plane.
        """
        self.dataframe = self.dataframe.dropna()
        pca = PCA(n_components=self.n_comp)
        pca_df = pca.fit_transform(self.dataframe)
        sns.set_theme(style="darkgrid")
        a_plot = sns.relplot(x=pca_df[:, 0],
                             y=pca_df[:, 1],
                             s=5)
        max_x = np.abs(max(pca_df[:, 0]))
        max_y = np.abs(max(pca_df[:, 1]))
        boundary = max(max_x, max_y) * 1.1
        a_plot.set(xlim=(-boundary, boundary))
        a_plot.set(ylim=(-boundary, boundary))
        return a_plot

    def scree(self):
        """
        Returns the proper values of a given dataframe.
        """
        pca = decomposition.PCA(n_components=self.n_comp)
        pca.fit(self.dataframe)
        return pca.singular_values_

    def feature_circle(self, pca, axis_ranks):
        """
        Display the correlations with arrows in the first factorial plane.
        """
        for dim_1, dim_2 in axis_ranks:
            if dim_2 < self.n_comp:
                _, axes = plt.subplots(figsize=(7, 6))
                # Features bars
                lines = [[[0, 0], [x, y]]
                         for x, y in self.dataframe[[dim_1, dim_2]].T]
                axes.add_collection(LineCollection(lines, axes=axes,
                                                 alpha=.1, color='black'))
                labels = self.dataframe['Label']
                # Variables names
                if labels:
                    for i, (x, y) in enumerate(self.dataframe[[dim_1,
                                                               dim_2]].T):
                        cond_1 = (x >= -1)
                        cond_2 = (x <= 1)
                        cond_3 = (y >= -1)
                        cond_4 = (y <= 1)
                        if cond_1 and cond_2 and cond_3 and cond_4:
                            plt.text(x, y, labels[i],
                                     fontsize='14', ha='center', va='center',
                                     color="blue", alpha=0.5)
                # Plot
                circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.plot([-1, 1], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-1, 1], color='grey', ls='--')
                ratio_x = pca.explained_variance_ratio_[dim_1]
                plt.xlabel('F{} ({}%)'.format(dim_1 + 1,
                                              round(100 * ratio_x, 1)))
                ratio_y = pca.explained_variance_ratio_[dim_2]
                plt.ylabel('F{} ({}%)'.format(dim_2 + 1,
                                              round(100 * ratio_y, 1)))
                title = "Cercle des corrélations (F{} et F{})"
                plt.title(title.format(dim_1+1, dim_2+1))
                plt.show(block=False)

    def circles(self, x_scaled):
        """
        Display the two correlations circles : samples & features.
        """
        pca = decomposition.PCA(n_components=self.n_comp)
        pca.fit(x_scaled)
        # pcs = pca.components_
        self.feature_circle(pca, [(0, 1)])
        plt.show()

    def df_proper_values(self):
        """
        Returns dataframe proper values after PCA reduction on n_comp
        components.
        """
        pca = decomposition.PCA(n_components=self.n_comp)
        pca.fit(self.dataframe)
        return pca.mean_

    def first_n_features_df(self):
        """
        Return the most significant columns of the original dataframe
        according to the PCA reduction.
        """
        if self.n_comp > self.dataframe.shape[1]:
            print('ERROR: New width greater than original dataframe width!')
            raise Exception()
        if self.n_comp > self.dataframe.shape[0]:
            print('WARNING: New width smaller than original dataframe length!')
        feature_means = self.df_proper_values()
        scree_df = pd.DataFrame({'feature': self.dataframe.columns,
                                 'mean': feature_means})
        scree_df = scree_df.sort_values(by='mean', ascending=False)
        best_scree_col = scree_df['feature'][:self.n_comp]
        return self.dataframe[best_scree_col]

    def pca_reduced_df(self):
        """
        Return PCA reduced dataframe.
        """
        pca = PCA(n_components=self.n_comp)
        new_df = pd.DataFrame(pca.fit_transform(self.dataframe),
                              index=self.dataframe.index)
        return new_df



class PerformancesEvaluator():
    """
    Help the performance evaluation of a PCA reduction.
    """
    def __init__(self, dataframe, n_comp):
        """
        Dataframe on which the PCA is performed, with the dimension to which it
        is reduced.
        """
        self.dataframe = dataframe
        self.n_comp = n_comp

    def perf_n_pca(self, pca_values_list, n_clust):
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
            pca_displayer = PcaDisplayer(self.dataframe, self.n_comp)
            pca_df = pca_displayer.first_n_features_df()
            # Apply k-means
            identifier = std_kmeans.KmeansIdentifier(pca_df)
            model = identifier.get_fitted_model(n_clust)
            [inertia, ca_ha, da_bo, sil] = identifier.get_kmeans_metrics(model)
            in_dict[pca_value] = inertia
            ch_dict[pca_value] = ca_ha
            db_dict[pca_value] = da_bo
            sil_dict[pca_value] = sil
        return [in_dict, ch_dict, db_dict, sil_dict]

    def perf_n_clust(self, n_clust_list):
        """
        Return metrics of PCA reduction performances on different number of
        principal components.
        """
        df_width = self.dataframe.shape[1]
        n_pca_list = [
            df_width,
            int(df_width*0.5), int(df_width*0.2), int(df_width*0.1),
            int(df_width*0.05), int(df_width*0.02), int(df_width*0.01)
            ]
        in_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        ch_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        db_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        sil_df = pd.DataFrame(columns=n_pca_list, index=n_clust_list)
        for n_clust in n_clust_list:
            print('Number of clusters:', n_clust)
            [inertia,
             calinski_harabasz,
             davies_bouldin,
             silhouette] = self.perf_n_pca(n_pca_list, n_clust)
            in_df.loc[n_clust] = list(inertia.values())
            ch_df.loc[n_clust] = list(calinski_harabasz.values())
            db_df.loc[n_clust] = list(davies_bouldin.values())
            sil_df.loc[n_clust] = list(silhouette.values())
        return in_df, ch_df, db_df, sil_df