from sklearn.metrics import silhouette_score
from src.cluster import get_cluster_directory, get_terms_path, get_tfidf_terms, get_k_means_cluster
from src.dump import read_file_to_list
from src.start import get_out_directory
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.metrics import silhouette_samples, silhouette_score


def load_label_and_terms(label_path, terms_path):
    labels = read_file_to_list(label_path)
    terms = read_file_to_list(terms_path)

    if len(labels) != len(terms):
        print "Data miss match"  + str(len(terms)) + ", " + str(len(labels))
        exit(1)

    X = []
    for index in range(0, len(labels)):
        obj = {}
        obj['Y'] = labels[index]
        obj['X'] = terms[index]
        X.append(obj)
    return X


def get_cluster(tfid_terms, num_cluster):
    X = tfid_terms['tfidf_matrix']

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    ax1.set_xlim([-0.3, 0.25])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, X._shape[0] + (num_cluster + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    km, cluster = get_k_means_cluster(X, num_cluster, is_list=False)
    sil_avg = silhouette_score(X, cluster)

    print "For number of clusters: " + str(num_cluster) + " average sil score:" + str(sil_avg)

  # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster)
    y_lower = 10
    for i in range(num_cluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / num_cluster)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=sil_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks

    # # 2nd Plot showing the actual clusters formed
    plt.show()


# this program is used find k clusters of data
if __name__ == "__main__":
    clister_dir = get_cluster_directory(get_out_directory())
    titles_path = get_out_directory() + "titles"
    X_Y_list = load_label_and_terms(titles_path, get_terms_path(clister_dir))

    X = []
    for obj in X_Y_list:
        X.append(obj['X'])
    tfidf_terms = get_tfidf_terms(X)
    for cluster in [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 20, 30]:
        get_cluster(tfidf_terms, cluster)

