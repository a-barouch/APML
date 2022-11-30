from sklearn.decomposition import PCA
import scipy.sparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
import matplotlib.patches as mpatches
from sklearn.manifold import LocallyLinearEmbedding
import pydiffmap

BEFORE_FILTERING = 'netflix_matrix'
AFTER_FILTERING = 'netflix_matrix_cleaned'
MOVIES_INFO_BEFORE_FILTERING = 'movies_info.pkl'
MOVIES_INFO_AFTER_FILTERING = 'movies_info_cleaned.pkl'


def save_files(mat_of_movies_and_users, df_of_movies_info):
    """
    This function saves the files which were created
    :param df_of_movies_info:
    :param mat_of_movies_and_users:
    :return:
    """
    try:
        print("Started saving pickle files")
        scipy.sparse.save_npz(AFTER_FILTERING, mat_of_movies_and_users.tocsr(), compressed=True)
        joblib.dump(df_of_movies_info, MOVIES_INFO_AFTER_FILTERING)
        print("Finished saving pickle files")
    except Exception as e:
        print("failed to save files")
        print(e)


def load_files(file_users, file_info):
    """
    This function loads both files from disk if they exist (one file is the ratings matrix and the other is the
    dataframe with the information about the movies)
    :return:
    """
    print("Started loading data from disk")
    mat_of_movies_and_users = scipy.sparse.load_npz(file_users + '.npz').tolil()
    df_of_movies_info = joblib.load(file_info)
    print("Finished loading data from disk")
    return df_of_movies_info, mat_of_movies_and_users


def dim_reduction(data, metadata, ALGORITHM):
    """
    reduce dimension by PCA/LLE/diffusion maps and then visualize the reduced data
    """

    if ALGORITHM == 0:
        # PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(data)
    elif ALGORITHM ==1:
        # LLE
        embedding = LocallyLinearEmbedding(n_components=2, n_neighbors=12)
        X_reduced = embedding.fit_transform(data)

    elif ALGORITHM ==2:
        # diffusion maps
        mydmap = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs = 2, epsilon = 'bgh', alpha = 0.001, k=20)
        mydmap.fit(data)
        X_reduced = mydmap.fit_transform(data)

    # if color by movie type
    for col in metadata.columns:
        idx = np.where(metadata[col] == 1)[0]
        # choose only movie types who appeared in more than X movies
        if (idx.size < 40):
            continue
        x_pca_col = np.take(X_reduced, idx, axis=0)
        plt.scatter(x_pca_col[:, 0], x_pca_col[:, 1], label=col)

        # add movie titles to some of the movies
        for i, txt in enumerate(np.take(metadata, idx, axis=0)['title']):
            if (np.random.rand(1) > 0.9):
                plt.annotate(txt, (X_reduced[:, 0][i], X_reduced[:, 1][i]), fontsize=6)

    # if color by year
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=metadata['year_of_release'], cmap='rainbow')
    # plt.colorbar()

    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.title('LLE Netflix data - k=12')
    plt.show()


def clustering(data, movies_info):
    """
    cluster LLE reduced data by spectral clustering
    """

    # reduce dimension by LLE
    embedding = LocallyLinearEmbedding(n_components=2, n_neighbors=12)
    X_principal = embedding.fit_transform(data)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']

    # compute spectral clustering
    spectral_model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')
    labels = spectral_model.fit_predict(X_principal)

    # plot by clusters
    color_key = {0: ('r'), 1: ('b'), 2: ('g'), 3: ('c')}
    patches = [mpatches.Patch(color=color, label=label) for label, color in color_key.items()]
    plt.scatter(X_principal['P1'], X_principal['P2'], marker='.', alpha=0.7, c=[color_key[index] for index in labels])
    plt.legend(handles=patches, labels=[label for label in color_key.keys()])
    plt.title("Spectral Clustering on LLE for k=4")

    # add movie titles to some of the movies
    for i, txt in enumerate(movies_info['title']):
        if np.random.rand(1) > 0.7:
            plt.annotate(txt, (X_principal['P1'][i], X_principal['P2'][i]), fontsize=7)
    plt.show()


def filter_rows_cols(mat_of_movies_and_users, df_of_movies_info):
    """
    filter to have only 5% most rated movies and 5% most rating users
    """
    # filter users
    cols_sum = np.diff(mat_of_movies_and_users.tocsc().indptr)
    perc_col = (np.percentile(cols_sum, 95))
    cols_list = np.argwhere(cols_sum > perc_col)
    cols_list_flat = [val for sublist in cols_list for val in sublist]
    x_new = scipy.sparse.lil_matrix(scipy.sparse.csr_matrix(mat_of_movies_and_users)[:, cols_list_flat])

    # filter movies
    rows_sum = np.diff(mat_of_movies_and_users.tocsr().indptr)
    perc_row = (np.percentile(rows_sum, 95))
    rows_list = np.argwhere(rows_sum > perc_row)
    rows_list_flat = [val for sublist in rows_list for val in sublist]
    x_new = scipy.sparse.lil_matrix(scipy.sparse.csr_matrix(x_new)[rows_list_flat, :])
    movies_new = df_of_movies_info.iloc[rows_list_flat, :]
    return x_new, movies_new


def main():
    df_of_movies_info, mat_of_movies_and_users = load_files(AFTER_FILTERING, MOVIES_INFO_AFTER_FILTERING)

    # if loading unfiltered files, filter and save them
    # x_new, movies_new = filter_rows_cols(mat_of_movies_and_users, df_of_movies_info)
    # save_files(x_new, movies_new)

    # choose algorithm: 0 for PCA, 1 for LLE, 2 for diffusion maps
    ALGORITHM = 0

    movies_users_arr = mat_of_movies_and_users.toarray()
    dim_reduction(movies_users_arr, df_of_movies_info, ALGORITHM)
    clustering(movies_users_arr, df_of_movies_info)


if __name__ == '__main__':
    main()
