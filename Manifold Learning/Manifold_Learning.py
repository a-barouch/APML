import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import pairwise
from sklearn.manifold import LocallyLinearEmbedding
from scipy.linalg import eigh


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()


def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets._samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels ** 0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.reshape(X[i, :], (d, d)))

    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels ** 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    # create distance matrix
    dist = pairwise.euclidean_distances(X)
    n = X.shape[0]
    H = np.identity(n) - np.ones((n, n)) * (1 / n)
    S = -0.5 * np.dot(np.dot(H, dist), H)

    # calculate eigenvectors and eigenvalues for S
    eig_val, eig_vec = np.linalg.eig(S)
    eig_vec_d = eig_vec[:, :d]
    eig_val_d = np.sqrt(eig_val[:d])

    # create reduced dimension matrix
    res = eig_val_d * eig_vec_d
    return res, eig_val


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    embedding = LocallyLinearEmbedding(n_components=d, n_neighbors=k)
    X_transformed = embedding.fit_transform(X)
    return X_transformed


def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps) .
    :return: Nxd reduced data matrix.
    '''

    # create affinity matrix by heat kernel
    heat_kernel = np.exp(-pairwise.euclidean_distances(X) ** 2 / sigma)
    r = np.sum(heat_kernel, axis=0)

    # create transition Markov matrix
    D_cor = np.diag((r) ** -0.5)
    As = np.matmul(D_cor, np.matmul(heat_kernel, D_cor))

    # sort eigenvalues and eigenvectors
    eig_val, eig_vec = eigh(As)
    idx = eig_val.argsort()[::-1]
    eig_vec = eig_vec[:, idx]
    eig_val = eig_val[idx]

    # choose d eigenvectors
    best_eig_vec = eig_vec[:, 1:d + 1]
    best_eig_val = eig_val[1:d + 1]

    # create results matrix
    res = best_eig_vec * best_eig_val ** t
    return res


def create_random_data(epsilon):
    """
        create random noised data of dimension NxN that sits on dimension Nxd
    """
    n, p, d = 500, 1000, 10

    # create an nxp matrix that sits on a nxd dimension
    gaus_mat = np.random.rand(n, d)
    zeros_mat = np.zeros((n, p - d))
    X = np.concatenate((gaus_mat, zeros_mat), axis=1)

    # create rotation matrix
    gaus_mat2 = np.random.rand(p, p)
    Q, R = np.linalg.qr(gaus_mat2)

    # create noise
    Z = np.random.normal(loc=0, scale=1, size=(n, p))

    return np.dot(X, Q) + epsilon * Z


def scree_plot(eig_val, noise):
    """
        produce a scree plot from eigenvalues
    """
    eig_val = [i for i in eig_val if i > 0.5]
    max_eig = min(50, len(eig_val))
    fig = plt.figure(figsize=(8, 5))
    sing_vals = np.arange(max_eig) + 1
    plt.plot(sing_vals, eig_val[:max_eig], 'ro-', linewidth=2)
    plt.title('Scree Plot for epsilon = ' + str(noise))
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    return fig


def get_data(DATASET):
    """
        get dataset data
    """
    # swissrole
    if DATASET == 0:
        data, color = datasets._samples_generator.make_swiss_roll(n_samples=2000)
        return data, color

    # faces
    if DATASET == 1:
        with open("faces.pickle", 'rb') as f:
            data = pickle.load(f)
            return data, None


def dim_reduct(data, ALGORITHM):
    """
        reduce data dimension by algorithm
    """
    if ALGORITHM == 0:
        res, _ = MDS(data, d=2)
    if ALGORITHM == 1:
        res = LLE(data, d=2, k=20)
    if ALGORITHM == 2:
        res = DiffusionMap(data, d=2, sigma=1, t=2)
    return res


def plot_results(DATASET, data, labels, old_data):
    """
        plot dimension reduction results for dataset
    """
    if DATASET == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Spectral)
        plt.show()
    if DATASET == 1:
        plot_with_images(data, old_data, "Faces plot").show()


if __name__ == '__main__':

    # choose dataset: 0 for swissrole, 1 for faces
    DATASET = 1

    # choose algorithm: 0 for MDS, 1 for LLE, 2 for diffusion maps
    ALGORITHM = 1

    data, labels = get_data(DATASET)
    data_reducted = dim_reduct(data, ALGORITHM)
    plot_results(DATASET, data_reducted, labels, data)

    # plot scree plot for MDS as function of noise
    epsilons = [0, 0.01, 0.1, 0.15, 1]
    for epsilon in epsilons:
        noised_data = create_random_data(epsilon)
        res, eigen_val = MDS(noised_data, d=2)
        scree_plot(eigen_val, epsilon).show()
