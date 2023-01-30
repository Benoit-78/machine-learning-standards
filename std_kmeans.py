"""
Author: B.Delorme
Creation date: 24th June 2021
Main objective: to provide a support for k-means clustering.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE



class KmeansIdentifier():
    def __init__(self, df):
        self.df = df

    def get_fitted_model(self, n_clust):
        model = KMeans(n_clusters=n_clust)
        X = self.df.to_numpy()
        model = model.fit(X)
        return model

    def get_labels(self, n_clust):
        """
        Given a dataframe, fit k-means on it and adds labels as a new feature.
        """
        model = self.get_fitted_model(n_clust)
        labels = model.labels_
        return labels

    def get_kmeans_metrics(self, model):
        labels = model.labels_
        inertia = model.inertia_
        calinski_harabasz = metrics.calinski_harabasz_score(self.df, labels)
        davies_bouldin = metrics.davies_bouldin_score(self.df, labels)
        silhouette = metrics.silhouette_score(self.df, labels)
        kmean_metrics = [inertia, calinski_harabasz, davies_bouldin, silhouette]
        return kmean_metrics

    def plot_kmeans_in_tsne(self, n_clust):
        """
        Represents the projection in t-SNE with k-means labels as colors.
        """
        if 'k-means label' not in list(self.df.columns):
            labels = self.get_labels(n_clust)
        tsne_res = TSNE(n_components=2, random_state=0).fit_transform(self.df)
        kmeans_labels = np.expand_dims(labels, axis=1)
        tsne_res_add = np.append(tsne_res, kmeans_labels, axis=1)
        # Plot
        plt.title('k-means groups in t-SNE plan')
        sns.scatterplot(x=tsne_res_add[:, 0],
                        y=tsne_res_add[:, 1],
                        hue=tsne_res_add[:, 2],
                        # palette=sns.hls_palette(n_clust,
                        #                         as_cmap=True),
                        legend='full',
                        s=5)

    def kmeans_on_feature(self, my_feature, n_clust):
        """
        Represent the disparities between k-means groups for one feature of the
        original dataframe.
        """
        if 'k-means_labels' not in list(self.df.columns):
            self.get_kmeans_df(n_clust)
        n_clust = self.df['k-means_labels'].nunique()
        # Séparation des groupes
        groupe_df_list = []
        group_df_len = {}
        for i in range(n_clust):
            groupe_df_list.append(self.df[self.df['k-means_labels']==i])
            group_df_len[i] = len(groupe_df_list[i])
        # Représentation des poids des différents groupes
        group_df_len = sorted(group_df_len.items(), key=lambda item:item[0],
                              reverse=True)
        plt.figure(figsize=(4, 12))
        plt.barh([str(element[0]) for element in group_df_len],
                 [element[1] for element in group_df_len],
                 edgecolor='k')
