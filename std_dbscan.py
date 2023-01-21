# Author: B.Delorme
# Creation date: 24/06/2021
# Objective: provide a support for DBSCAN clustering

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from sklearn.manifold import TSNE


class DbscanClassifier():
    def __init__(self, eps=0.2, min_samples=2, leaf_size=20):
        self.eps = eps
        self.min_samples=min_samples
        self.leaf_size = leaf_size

    def dbscan_dataframe(self, df):
        """
        1) Fit dbscan on the given dataframe
        2) Adds the labels as a new feature.
        """
        dbscan_clustering = DBSCAN(eps=0.2, min_samples=2,leaf_size=20).fit(df)
        labels = dbscan_clustering.labels_
        # Special case of only noise points
        if len(Counter(labels)) == 1:
            return print('Only noise points')
        ch = metrics.calinski_harabasz_score(df, labels)
        db = metrics.davies_bouldin_score(df, labels)
        sil = metrics.silhouette_score(df, labels)
        df['DBSCAN label'] = labels
        return df, ch, db, sil

    def plot_dbscan_in_tsne(self, df):
        """
        Projection in t-SNE with dbscan labels as colors.
        """
        # has df already been processed through dbscan?
        if 'DBSCAN label' not in list(df.columns):
            my_dataframe, a, b, c = self.dbscan_dataframe(df)
        # Number of clusters in labels, ignoring noise if present.
        n_clust = len(set(df['DBSCAN label'])) - (1 if -1 in df['DBSCAN label'] else 0)
        # Count cluster sizes
        unique, counts = np.unique(df['DBSCAN label'], return_counts=True)
        my_dict = dict(zip(unique, counts))
        tsne_df = my_dataframe.drop('DBSCAN label', axis=1)
        # t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        tsne_res = tsne.fit_transform(tsne_df)
        dbscan_lab = np.expand_dims(my_dataframe['DBSCAN label'], axis=1)
        # Add k-means labels to t-SNE coordinates
        tsne_res_add = np.append(tsne_res, dbscan_lab, axis=1)
        # Plot
        n_dim = len(list(my_dataframe.columns)) - 1
        plt.title('DBSCAN groups in t-SNE plan for {} principal components'.format(n_dim))
        sns.scatterplot(x=tsne_res_add[:, 0],
                        y=tsne_res_add[:, 1],
                        hue=tsne_res_add[:, 2], palette=sns.hls_palette(n_clust), legend='full', s=5)

    def evaluate_dbscan(self, df, n_pca):
        """
        Returns three performance indicators of dbscan:
        1) the amount of noise,
        2) the size of the bigger group,
        3) the average size of the other groups.
        """
        dbscan_clustering = DBSCAN().fit(pca_dataframe(df, 21))
        labels = dbscan_clustering.labels_
        temp_dict = dict(Counter(labels))
        n_noise = temp_dict[-1]
        n_big = temp_dict[0]
        temp_dict.pop(-1)
        temp_dict.pop(0)
        temp_list = list(temp_dict.values())
        average = sum(temp_list) / len(temp_list)
        return n_noise, n_big, average