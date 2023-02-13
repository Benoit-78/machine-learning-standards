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
        self.dataframe = dataframe
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
        model.fit(self.dataframe)
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
        self.dataframe['DBSCAN label'] = labels

    def plot_dbscan_in_tsne(self):
        """
        Projection in t-SNE with dbscan labels as colors.
        """
        if 'DBSCAN label' not in list(self.dataframe.columns):
            self.add_labels()
        # Cluster sizes
        labels = self.dataframe['DBSCAN label'].nunique()
        n_clust = labels - (1 if -1 in self.dataframe['DBSCAN label'] else 0)
        # t-SNE
        tsne_df = self.dataframe.drop('DBSCAN label', axis=1)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_res = tsne.fit_transform(tsne_df)
        dbscan_lab = np.expand_dims(self.dataframe['DBSCAN label'], axis=1)
        tsne_res_add = np.append(tsne_res, dbscan_lab, axis=1)
        # Plot
        plt.title('DBSCAN groups in t-SNE plan')
        sns.scatterplot(x=tsne_res_add[:, 0],
                        y=tsne_res_add[:, 1],
                        hue=tsne_res_add[:, 2],
                        palette=sns.hls_palette(n_clust, as_cmap=True),
                        legend='full', s=5)

    def get_standard_metrics(self):
        """
        Get standard metrics.
        Some of them will be used for computation of a cost function.
        """
        model = self.get_fitted_model()
        labels = model.labels_
        # Number of samples
        n_samples = self.dataframe.shape[0]
        # Number of groups
        labels_counter = dict(Counter(labels))
        n_groups = len(labels_counter)
        # Number of noise points
        if -1 in labels_counter.keys():
            n_noise = labels_counter[-1]
            labels_counter.pop(-1)
        else:
            n_noise = "No noise point."
        # Biggest group size
        n_biggest = max(labels_counter.values())
        # Average group size
        counts = list(labels_counter.values())
        average = sum(counts) / len(counts)
        average = int(average)
        # Davies-Bouldin metrics
        temp_df = self.dataframe[self.dataframe['DBSCAN label']!=-1]
        davies_bouldin = davies_bouldin_score(temp_df, temp_df['DBSCAN label'])
        davies_bouldin = round(davies_bouldin, 2)
        # Result
        metrics_dict = {"Number of samples": n_samples,
                        "Number of groups": n_groups,
                        "Number of noise points": n_noise,
                        "Biggest group size": n_biggest,
                        "Average group size": average,
                        "Davies-Boudin metrics": davies_bouldin}
        return metrics_dict
