"""
Author: B.Delorme
Mail: delormebenoit211@gmail.com
Creation date: 24/06/2021
Objective: provide a support for DBSCAN clustering
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


class DbscanClassifier():
    """
    Provides methods to identify groups in clustering tasks.
    """
    def __init__(self, dataframe, eps=0.2, min_samples=2, leaf_size=20):
        """
        df : pandas DataFrame
            Data table
        eps : float
            - The maximum distance between two samples for one to be considered
              as in the neighborhood of the other.
            - The default is 0.2.
        min_samples : int
            - The number of samples (or total weight) in a neighborhood for a
              point to be considered as a core point.
            - The default is 2.
        leaf_size : int
            - Leaf size passed to BallTree or cKDTree.
            - The default is 20.
        """
        self.df = dataframe
        self.eps = eps
        self.min_samples=min_samples
        self.leaf_size = leaf_size

    def get_fitted_model(self):
        """
        Fit dbscan on the given dataframe.
        """
        model = DBSCAN(eps=self.eps,
                       min_samples=self.min_samples,
                       leaf_size=self.leaf_size)
        model.fit(self.df)
        return model

    def add_labels(self):
        """
        Add the labels as a new feature.
        """
        model = self.get_fitted_model()
        labels = model.labels_
        # Special case of only noise points
        if len(Counter(labels)) == 1:
            print('Only noise points')
        self.df['DBSCAN label'] = labels

    def plot_dbscan_in_tsne(self):
        """
        Projection in t-SNE with dbscan labels as colors.
        """
        if 'DBSCAN label' not in list(self.df.columns):
            self.add_labels()
        # Cluster sizes
        labels = self.df['DBSCAN label'].nunique()
        n_clust = labels - (1 if -1 in self.df['DBSCAN label'] else 0)
        # t-SNE
        tsne_df = self.df.drop('DBSCAN label', axis=1)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_res = tsne.fit_transform(tsne_df)
        dbscan_lab = np.expand_dims(self.df['DBSCAN label'], axis=1)
        tsne_res_add = np.append(tsne_res, dbscan_lab, axis=1)
        # Plot
        plt.title('DBSCAN groups in t-SNE plan')
        sns.scatterplot(x=tsne_res_add[:, 0],
                        y=tsne_res_add[:, 1],
                        hue=tsne_res_add[:, 2],
                        palette=sns.hls_palette(n_clust, as_cmap=True),
                        legend='full', s=5)

    def get_metrics(self):
        """
        Performance indicators.
        """
        model = self.get_fitted_model()
        labels = model.labels_
        labels_counter = dict(Counter(labels))
        n_noise = labels_counter[-1]
        n_big = max(labels_counter.values())
        labels_counter.pop(-1)
        counts = list(labels_counter.values())
        average = sum(counts) / len(counts)
        average = round(average, 1)
        metrics_dict = {"Number of groups": len(labels_counter),
                        "Number of noise points": n_noise,
                        "Bigger group size": n_big,
                        "Average group size": average}
        return metrics_dict
