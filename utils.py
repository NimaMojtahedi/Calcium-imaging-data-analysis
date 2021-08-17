# In this file we are providing usefull functions

# important libraries
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import pandas as pd
import seaborn as sns
import pdb
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from matplotlib import animation
from IPython.display import HTML

# functions and classes


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


class NMFCV:
    """
    This class provides some methods for running nmf and estimating true number of components usign cross validation
    INPUT
    data = input data n*p matrix
    use_feature_selection = using variance based feature selection if threshold is not given program ask user for threshold
    """

    def __init__(self, data, use_feature_selection=False, threshold=False):

        self.data = data
        self.use_feature_selection = use_feature_selection
        self.threshold = threshold

    def apply_feature_selection(self):
        from sklearn.feature_selection import VarianceThreshold

        n, p = self.data.shape

        if self.threshold:
            # initialize class with user given threshold
            var_feature = VarianceThreshold(threshold=self.threshold)

        else:
            var_feature = VarianceThreshold(
                threshold=int(input('please give threshold!')))

        # fit on data
        var_feature.fit(self.data)

        # transform data
        feature_selected = var_feature.transform(self.data)
        print(
            f'selected feature shape for given threshold: {feature_selected.shape}')

        # save to self
        self.var_feature = var_feature
        self.feature_selected = feature_selected

        return feature_selected

    def estimate_componentsV2(self, n_jobs=5, nr_replicates=5, nr_components=[5, 80, 5], n_iters=2000, mask_portion=20):
        # estimating using sklearn package by masking data randomly
        # main data
        if self.use_feature_selection:
            X_train = self.apply_feature_selection()
        else:
            X_train = self.data  # n*p

        # data size
        n, p = X_train.shape

        # zero indices
        z_cols = np.random.randint(0, p-1, int(n*p*mask_portion/100))
        z_rows = np.random.randint(0, n-1, int(n*p*mask_portion/100))

        # prepare sparse matrix
        train = X_train.copy()
        train[z_rows, z_cols] = 0

        ix = np.nonzero(train)
        sparse_mat = sparse.csc_matrix((train[ix], ix))

        # run nmf in parallel
        results = Parallel(n_jobs=n_jobs)(delayed(self.estimate_componentsV2_helper)(X_train, sparse_mat, n_cmp, n_rep,
                                                                                     z_rows, z_cols, n_iters) for n_cmp in range(nr_components[0],
                                                                                                                                 nr_components[
                                                                                                                                     1],
                                                                                                                                 nr_components[2]) for n_rep in range(nr_replicates))
        self.nmf_cv_results = results
        return results

    def estimate_componentsV2_helper(self, data, train_sparse, nr_cmp, rep_nr, z_rows, z_cols, n_iters=2000):
        # train is sparce matrix
        # test is normal matrix

        # model fitting
        model = NMF(n_components=nr_cmp, init="nndsvd",
                    max_iter=n_iters).fit(train_sparse)
        reconstructed = model.inverse_transform(model.transform(train_sparse))

        # error calculation
        # train
        mse_train = mean_squared_error(
            data[~z_rows, ~z_cols], reconstructed[~z_rows, ~z_cols], multioutput='uniform_average')
        r2e_train = r2_score(
            data[~z_rows, ~z_cols], reconstructed[~z_rows, ~z_cols], multioutput='uniform_average')
        evar_train = explained_variance_score(
            data[~z_rows, ~z_cols], reconstructed[~z_rows, ~z_cols], multioutput='uniform_average')

        # test
        mse_test = mean_squared_error(
            data[z_rows, z_cols], reconstructed[z_rows, z_cols], multioutput='uniform_average')
        r2e_test = r2_score(
            data[z_rows, z_cols], reconstructed[z_rows, z_cols], multioutput='uniform_average')
        evar_test = explained_variance_score(
            data[z_rows, z_cols], reconstructed[z_rows, z_cols], multioutput='uniform_average')

        return nr_cmp, rep_nr, mse_train, r2e_train, evar_train, mse_test, r2e_test, evar_test

    def estimate_componentsV1(self, use_parallel=False, n_jobs=1, nr_replicates=5, nr_components=[5, 80, 5], n_iters=50):
        # estimating number of components in simple for loop or using parallel backend

        # feature reduction
        if self.use_feature_selection:
            data = self.apply_feature_selection()

        else:
            data = self.data
        if use_parallel:
            # run in parallel
            start_time = time.clock()
            nmf_cv_results = Parallel(n_jobs=n_jobs, verbose=10,
                                      backend='loky')(delayed(self.NMF_CV)(data=data,
                                                                           rank=i,
                                                                           replicates=j,
                                                                           nr_iter=n_iters) for j in range(nr_replicates) for i in range(nr_components[0],
                                                                                                                                         nr_components[
                                                                                                                                             1],
                                                                                                                                         nr_components[2]))
            print(
                f'execution time: {np.rint(time.clock() - start_time)} seconds')

            self.nmf_cv_results = nmf_cv_results
            return nmf_cv_results
        else:
            # initialize
            nmf_cv_results = []

            # run NMF_CV normal
            start_time = time.clock()
            nmf_cv_results = self.NMF_CV_loop(data=data, rank_range=np.arange(
                nr_components[0], nr_components[1], nr_components[2]), replicates=nr_replicates, nr_iter=n_iters)
            print(
                f'execution time: {np.rint(time.clock() - start_time)} seconds')

            self.nmf_cv_results = nmf_cv_results
            return nmf_cv_results

    def NMF_CV_loop(self, data, rank_range, replicates, nr_iter=50):
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
        results = []

        # run loop
        for rnk, rep in tqdm(itertools.product(rank_range, range(replicates))):
            tr, te = cv_pca(data, rnk, nonneg=True, nr_iter=nr_iter)[2:]
            results.append((rnk, rep, tr, te))

            #  printing each loop results
            print(
                f'Replication {rep}, number of components {rnk}, train_error: {tr} - test_error: {te}')

        return results

    def NMF_CV(self, data, rank, replicates, nr_iter=50):
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

        # run nmf_cv
        tr_error, te_error = cv_pca(
            data, rank, nonneg=True, nr_iter=nr_iter)[2:]

        return rank, replicates, tr_error, te_error

    def run_nmf(self, nr_components):
        from sklearn.decomposition import NMF

        sk_nmf = NMF(n_components=nr_components, random_state=1)

        # check data type
        if self.use_feature_selection:

            # fitting and getting transformation
            traces = sk_nmf.fit_transform(self.feature_selected)

            # getting components
            temp_cmp = sk_nmf.components_

            # returning abck components to original space
            components = self.var_feature.inverse_transform(temp_cmp)

        else:
            # fitting and getting transformation
            traces = sk_nmf.fit_transform(self.data)

            # getting components
            components = sk_nmf.components_

        return components, traces

    def plot_cv_results(self, columns, min_by, save_fig_add=False):

        cv_results = pd.DataFrame(self.nmf_cv_results, columns=columns)

        # change to long format
        cv_results = cv_results.melt(id_vars=['Components', 'Replication'],
                                     var_name='Error_Type', value_name='Error')

        # log error
        cv_results['Error(log)'] = cv_results['Error'].apply(
            lambda x: np.log(x))

        # finding best component number
        min_index = cv_results[cv_results.Error_Type == min_by].groupby(
            by=['Components']).agg(np.mean)['Error'].idxmin()
        print(f'Component number with minimum test error is {min_index}')

        # ploting
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        sns.lineplot(data=cv_results, x="Components",
                     y="Error(log)", hue='Error_Type', marker='*', ax=ax, palette='bone_r')
        ax.axvline(x=min_index, alpha=0.5, color='blue', linewidth=2,
                   linestyle='-.', label='Optimal component numbers')
        ax.legend()

        # saving figure
        if save_fig_add:
            fig.savefig(fname=save_fig_add + '/nmf_cv_result.pdf',
                        dpi=600, quality=100, format='pdf')


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


def video_player(np_array_video, intervals_=50):

    # np array with shape (frames, height, width, channels)
    if len(np_array_video.shape) == 3:
        np_array_video = np_array_video[..., np.newaxis]

    video = np.array(np_array_video)

    fig = plt.figure()
    im = plt.imshow(video[0, :, :, :])

    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(video[0, :, :, :])

    def animate(i):
        im.set_data(video[i, :, :, :])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                   interval=intervals_)
    return HTML(anim.to_html5_video())


# can calculate connectivity in specific frequency (principal frequencies)
# filtering lowpass, high pass, bandpass, ...
# CCA
