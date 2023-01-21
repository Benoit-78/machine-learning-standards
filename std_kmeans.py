# Author: B.Delorme
# Creation date: 24/06/2021
# Main objective: to provide a support for k-means clustering.


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from sklearn.manifold import TSNE



class KmeansClassifier():
    def kmeans_dataframe(self, df, n_clust, n_init=10, max_iter=300, tol=0.0001):
        """Given a dataframe, fit k-means on it and adds the labels as a new feature."""
        kmeans_model = KMeans(n_clusters= n_clust,
                              n_init=n_init,
                              max_iter=max_iter,
                              tol=tol).fit(df)
        labels = kmeans_model.labels_
        #inertia = kmeans_model.inertia_
        #calinski_harabasz = metrics.calinski_harabasz_score(df, labels)
        #davies_bouldin = metrics.davies_bouldin_score(df, labels)
        #silhouette = metrics.silhouette_score(df, labels)
        df['k-means label'] = labels
        return df#, inertia, calinski_harabasz, davies_bouldin, silhouette

    def plot_kmeans_in_tsne(self, df, n_clust):
        """Represents the projection in t-SNE with k-means labels as colors."""
        # has df already been processed through k-means?
        if 'k-means label' not in list(df.columns):
            df = self.kmeans_dataframe(df, n_clust)
        tsne_df = df.drop('k-means label', axis=1)
        tsne_res = TSNE(n_components=2, random_state=0).fit_transform(tsne_df)
        # Increase of one dimension
        kmeans_lab = np.expand_dims(df['k-means label'], axis=1)
        # Add k-means labels to t-SNE coordinates
        tsne_res_add = np.append(tsne_res, kmeans_lab, axis=1)
        # Plot
        n_dim = df.shape[1] - 1
        plt.title('k-means groups in t-SNE plan for {} principal components'.format(n_dim))
        sns.scatterplot(x=tsne_res_add[:, 0], y=tsne_res_add[:, 1],
                        hue=tsne_res_add[:, 2], palette=sns.hls_palette(n_clust),
                        legend='full', s=5)

    def kmeans_on_feature(self, my_dataframe, my_feature):
        """Represent the disparities between k-means groups for one feature of the
        original dataframe.
        """
        # Check that the given dataframe has been processed through k-means
        if 'k-means_labels' not in list(my_pca_dataframe.columns):
            return print('No k-means label column in the given pca dataframe.')
        n_clust = my_dataframe['k-means_labels'].nunique()
        # Séparation des groupes
        groupe_df_list = []
        group_df_len = {}
        for i in range(n_clust):
            groupe_df_list.append(temp_labels_df[temp_labels_df['k-means_labels']==i])
            group_df_len[i] = len(groupe_df_list[i])
        # Représentation des poids des différents groupes
        group_df_len = sorted(group_df_len.items(),
                              key=lambda item:item[0],
                              reverse=True)
        plt.figure(figsize=(4, 12))
        plt.barh([str(element[0]) for element in group_df_len],
                 [element[1] for element in group_df_len],
                 edgecolor='k')