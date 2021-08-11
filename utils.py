# In this file we are providing usefull functions


class DeltaF:

    """
    class calculates deltaf / f0 for a given trace
    To select f0 (basedline), 0-50 percentile of data is selected as given to robust
    regression estimtor

    INPUT
    data: n*p matrix; n length of trace and p number of traces
    prct: lower percentile threshold
    """
    # import libraries for class
    import numpy as np
    from sklearn import linear_model
    from tqdm import tqdm

    def __init__(self, data, prct):

        # initialize class
        self.data = data

        # get data dimension
        n, p = data.shape
        self.n = n
        self.p = p
        self.prct = prct

    def fit_transform(self, X, y, X_pr):

        # define RANSAC model
        ransac = self.linear_model.RANSACRegressor()

        # fit ransac
        ransac.fit(X, y)

        # predict for new data
        prdct = ransac.predict(X_pr)

        return prdct

    def run_ransac(self, sig):

        # running ransac fit/predict on input signal

        # find indices
        index_ = self.find_index(X=sig)

        # fit predict ransac
        base_line = self.fit_transform(X=index_.reshape(-1, 1),
                                       y=sig[index_],
                                       X_pr=self.np.arange(len(sig)).reshape(-1, 1))

        return base_line

    def find_index(self, X):

        # finding indices between 0 and 50 percentile
        prct = self.np.percentile(X, self.prct)

        return self.np.squeeze(self.np.where(X < prct))

    def run_on_matrix(self):

        # this function run ransac on matrix input n * p

        # initialize result matrix
        result = self.np.zeros_like(self.data)

        # run ransac
        for i in self.tqdm(range(self.p)):
            result[:, i] = self.run_ransac(sig=self.data[:, i])

        # result = [self.run_ransac(sig=self.data[:, i]) for i in range(self.p)]
        # result = self.np.stack(result).T

        # calculate deltaf/f0
        result = (self.data - result) / result

        return result


# data loader function
def load_file(add, key_name=False):
    """
    load_file function reads data in h5 format
    INPUT
    add: file location
    """
    # import libraries
    import h5py as h5
    import numpy as np

    # reading file
    file = h5.File(add)

    if not key_name:
        # printing keys
        for key in file.keys():
            print(key)

        # ask from user for proper key
        key_name = input('Please input proper key name!')

    # take user data
    data = file[key_name]

    # create numpy float16 array from data
    data = np.array(data, dtype=np.float32)

    # data shape info
    print(f'selected file size: {data.shape}')

    return data


class DataResize:
    """
    this class takes data in 3D (n * p * p'; n: number of frames in time, p * p': frame size)
    """
    import cv2
    import numpy as np

    def __init__(self, data, dim=(100, 100)):

        self.data = data
        self.dim = dim

    def frame_resize(self, img, dim=(100, 100)):
        """
        frame_resize function get input image and resize to given dim (tuple)
        INPUTs
        img: n * p matrix (can have 3 dimension  as well)
        dim: new dimension
        """

        # resizing
        img_new = self.cv2.resize(
            img, dim, interpolation=self.cv2.INTER_AREA)

        return img_new

    def transform(self):

        # data needs to be in 3D

        data_new = [self.frame_resize(self.data[i, :, :], dim=self.dim)
                    for i in range(self.data.shape[0])]

        return self.np.stack(data_new)


def NMF_CV_loop(data, rank_range, replicates, nr_iter=50):
    """
    fitting nmf model with cross-validation on component numbers
    INPUTs
    data: input data n * p matrix
    rank_range: list of numbers for number of component parameter
    replicates: number of replicates int
    """
    # import nmf_cv
    from cv import cv_pca
    import itertools
    from tqdm import tqdm

    # initialize train/test errors
    train_err, test_err = [], []

    # run loop
    for rnk, rep in tqdm(itertools.product(rank_range, range(replicates))):
        tr, te = cv_pca(data, rnk, nonneg=True, nr_iter=nr_iter)[2:]
        train_err.append((rep, rnk, tr))
        test_err.append((rep, rnk, te))

        #  printing each loop results
        print(
            f'Replication {rep}, number of components {rnk}, train_error: {tr} - test_error: {te}')

    return train_err, test_err


def NMF_CV(data, rank, replicates, nr_iter=50):
    """
    fitting nmf model with cross-validation on component numbers
    INPUTs
    data: input data n * p matrix
    rank: int
    replicates: number of replicates int
    This is good for parallel calculation
    """
    # import nmf_cv
    from cv import cv_pca
    import itertools
    from tqdm import tqdm

    # run nmf_cv
    tr_error, te_error = cv_pca(data, rank, nonneg=True, nr_iter=nr_iter)[2:]
    print(
        f'Replication {replicates}, number of components {rank}, train_error: {tr_error} - test_error: {te_error}')

    return rank, replicates, tr_error, te_error


class Connectivity:
    """
    In this class we investigate connectivity and network parameters for given data
    This class has several methods for different tasks. 
    Preprocessing method takes 3D tensor and subsamples spatially the data with given resolution to give row/col location and traces in time
    if labels are not given user can define per location label name using get_label method
    Using covariance_lasso we estimate sparse covariance and precision matrices with auto tune sparsity paramter using 5 fold cross validation

    INPUT
    data: n * p * p tensor
    """
    # import libraries
    from sklearn.covariance import GraphicalLassoCV
    from matplotlib import pyplot as plt
    import numpy as np
    # import mne
    import networkx as nx

    # initialize class and get input tensor
    def __init__(self, data, resolution=5, labels=False):
        self.data = data
        self.n = data.shape[0]
        self.nr_row = data.shape[1]
        self.nr_col = data.shape[2]
        self.resolution = resolution
        self.labels = labels

    # apply preprocessing step - spatially subsample data and store location and values in list
    def preprocessing(self):
        # get tensor data
        data = self.data
        nr_row = self.nr_row
        nr_col = self.nr_col
        resolution = self.resolution

        # initialize location value list
        loc_val = []

        # get location and traces
        for i in range(0, nr_col, resolution):
            for j in range(0, nr_row, resolution):
                loc_val.append([j, i, self.data[:, j, i]])

        self.location_value = loc_val

        # if labels are not provided
        if not self.labels:
            self.labels = self.get_labels()

        return loc_val

    # if labels are not provided get labels per location
    def get_labels(self):

        # reading locations from preprocessing function and asking user for label information
        loc_val = self.location_value

        # getting data and prepare average frame
        img = self.np.mean(self.data, axis=0)

        # initialize figure
        fig, ax = self.plt.subplots(1, 1, figsize=(5, 5))

        # initialize label list
        labels = []

        # ask labels
        for item in loc_val:
            ax.imshow(img)
            ax.plot(item[0], item[1], 'o', ms=5)
            labels.append(input('please write label name!'))

        self.labels = labels
        return labels

    # after providing labels name, saving results
    def save_labels(self, address):
        self.np.save(address + '\labels', self.labels)

    # estimate sparse covariance matrix
    def covariance_lasso(self, alpha=10, max_iter=200, mode='cd', n_jobs=-1):

        # get location value from preprocessing
        loc_val = self.preprocessing()

        # prepare data
        data = [item[2] for item in loc_val]
        data = self.np.stack(data)

        # fitting model
        cov = self.GraphLassoCV(
            alpha=alpha, max_iter=max_iter, mode=mode, n_jobs=n_jobs).fit(data)

        self.model = cov
        return cov

    # plotting sparsity alpha value
    def plot_model_selection(self):
        model = self.model
        self.plt.figure(figsize=(4, 3))
        self.plt.axes([.2, .15, .75, .7])
        self.plt.plot(model.cv_results_["alphas"],
                      model.cv_results_["mean_score"], 'o-')
        self.plt.axvline(model.alpha_, color='.5')
        self.plt.title('Model selection')
        self.plt.ylabel('Cross-validation score')
        self.plt.xlabel('alpha')

        self.plt.show()

    # plot graph
    def plot_network_graph(self, adj_matrix, title):

        # load labels and loc_val
        loc = self.location_value
        labels = self.labels

        node_dict = {}
        # create node dictionary
        for i in range(len(labels)):
            node_dict.update({labels[i]: loc[i][0]})

        # creating graph from adj_matrix
        graph = self.nx.from_numpy_matrix(adj_matrix)

        self.plt.show()

# can calculate connectivity in specific frequency (principal frequencies)
# filtering lowpass, high pass, bandpass, ...
