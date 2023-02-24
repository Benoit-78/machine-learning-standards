"""
Author: B.Delorme
Mail: delormebenoit211@gmail.com
Creation date: 24/06/2021
Objective: provide a support for DBSCAN clustering
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score


class DbscanClassifier():
    """
    Provides methods to identify groups in clustering tasks.
    """
    def __init__(self, dataframe, eps=2, min_samples=2, leaf_size=20):
        """
        eps : maximum distance between two samples for one to be considered
              as in the neighborhood of the other.
        min_samples : number of samples (or total weight) in a neighborhood for
                    a point to be considered as a core point.
        leaf_size : leaf size passed to BallTree or cKDTree.
        """
        self.dataframe = dataframe.copy()
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size

    def get_fitted_model(self):
        """
        Fit DBSCAN on the given dataframe.
        """
        model = DBSCAN(eps=self.eps,
                       min_samples=self.min_samples,
                       leaf_size=self.leaf_size,
                       n_jobs=-1)
        model.fit(self.dataframe)
        return model

    def add_labels(self):
        """
        Add the labels as a new feature.
        """
        model = self.get_fitted_model()
        labels = model.labels_
        # Special case of only noise points
        only_one_label = (pd.Series(labels).nunique() == 1)
        noise_points = (pd.Series(labels).unique()[0] == -1)
        if only_one_label and noise_points:
            print('Only noise points')
        self.dataframe['DBSCAN label'] = pd.Series(labels)

    def plot_dbscan_in_tsne(self):
        """
        Projection in t-SNE with dbscan labels as colors.
        """
        model = self.get_fitted_model()
        # Cluster sizes
        labels = pd.Series(model.labels_)
        noise = (1 if -1 in list(labels) else 0)
        n_clusters = labels.nunique() - noise
        # t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        tsne_res = tsne.fit_transform(self.dataframe)
        dbscan_lab = np.expand_dims(labels, axis=1)
        tsne_res_add = np.append(tsne_res, dbscan_lab, axis=1)
        # Plot
        plt.figure(figsize=(8, 8))
        plt.title('{} DBSCAN clusters in t-SNE plan'.format(n_clusters))
        sns.scatterplot(x=tsne_res_add[:, 0],
                        y=tsne_res_add[:, 1],
                        hue=tsne_res_add[:, 2], s=80, edgecolors='black',
                        alpha=0.7,
                        palette=sns.hls_palette(n_clusters, as_cmap=True))
        plt.axis('square')

    def get_standard_metrics(self):
        """
        Get standard metrics.
        Some of them will be used for computation of a cost function.
        """
        model = self.get_fitted_model()
        if 'DBSCAN label' not in list(self.dataframe.columns):
            self.add_labels()
        labels = pd.Series(model.labels_)
        labels_counter = dict(Counter(labels))
        # Number of noise points
        if -1 in labels_counter.keys():
            n_noise = labels_counter[-1]
            labels_counter.pop(-1)
        else:
            print("INFO: No noise point.")
            n_noise = 0
        # Biggest group size
        n_biggest = max(labels_counter.values())
        # Average group size
        n_clusters = len(labels_counter)
        average = sum(labels_counter.values()) / n_clusters
        average = int(average)
        # Davies-Bouldin metrics
        noise_free_df = self.dataframe[self.dataframe['DBSCAN label']!=-1]
        if noise_free_df['DBSCAN label'].nunique()==1:
            print('WARNING: Only one cluster. Davies-Bouldin score set to 1.')
            davies_bouldin = 1
        elif noise_free_df.shape[0] > 1:
            davies_bouldin = davies_bouldin_score(noise_free_df,
                                                  noise_free_df['DBSCAN label']
                                                  )
            davies_bouldin = round(davies_bouldin, 2)
        else:
            print('WARNING: Only noise points. Davies-Bouldin score set to 1.')
            davies_bouldin = 1
        # Result
        metrics_dict = {"n_samples": self.dataframe.shape[0],
                        "n_clusters": n_clusters,
                        "n_noise": n_noise,
                        "n_biggest": n_biggest,
                        "average": average,
                        "davies_bouldin": davies_bouldin}
        return metrics_dict
