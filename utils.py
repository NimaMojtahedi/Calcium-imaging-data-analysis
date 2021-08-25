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
import nimfa
from skimage import morphology
from skimage import measure
from mne.viz import circular_layout
from mne.viz import plot_connectivity_circle
from sklearn.covariance import GraphicalLassoCV

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
        # index_ = self.find_index(X=sig) when data has trend it causes problem
        # to solve trend I use all data set at the moment
        index_ = np.arange(len(sig))

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

    def estimate_componentsV3(self, nmf_type, rank_cands=range(5, 30, 3), max_iter=200):
        """
        nmf_type : Types of nmfs
                    - nmf
                    - bd
                    - Icm
                    - Lfnmf
                    - Nsnmf
                    - PMF
                    - Pmfcc
        """

        # main data
        if self.use_feature_selection:
            V = self.apply_feature_selection()
        else:
            V = self.data  # n*p

        # check type
        if nmf_type.lower() == "nmf":
            nmf = nimfa.Nmf(V, seed="nndsvd", rank=10, max_iter=max_iter, update='euclidean',
                            objective='fro')
            summary = nmf.estimate_rank(rank_range=rank_cands)

        elif nmf_type.lower() == "bd":
            nmf = nimfa.Bd(V, seed="nndsvd", rank=10, max_iter=max_iter, alpha=np.zeros((V.shape[0], 10)),
                           beta=np.zeros((10, V.shape[1])), theta=.0, k=.0, sigma=1., skip=1000, stride=1,
                           n_w=np.zeros((10, 1)), n_h=np.zeros((10, 1)), n_sigma=False)
            summary = nmf.estimate_rank(rank_range=rank_cands)

        elif nmf_type.lower() == "icm":
            nmf = nimfa.Icm(V, seed="nndsvd", rank=10, max_iter=max_iter, iiter=20,
                            alpha=np.random.randn(V.shape[0], 10), beta=np.random.randn(10, V.shape[1]),
                            theta=0., k=0., sigma=1.)
            summary = nmf.estimate_rank(rank_range=rank_cands)

        elif nmf_type.lower() == "lfnmf":
            nmf = nimfa.Lfnmf(V, seed="nndsvd", W=np.random.rand(V.shape[0], 10),
                              H=np.random.rand(10, V.shape[1]), rank=10, max_iter=max_iter,
                              alpha=0.01)
            summary = nmf.estimate_rank(rank_range=rank_cands)

        elif nmf_type.lower() == "nsnmf":
            nmf = nimfa.Nsnmf(V, seed="nndsvd", rank=10, max_iter=max_iter)
            summary = nmf.estimate_rank(rank_range=rank_cands)

        elif nmf_type.lower() == "pmf":
            nmf = nimfa.Pmf(V, seed="nndsvd", rank=10,
                            max_iter=max_iter, rel_error=1e-5)
            summary = nmf.estimate_rank(rank_range=rank_cands)

        elif nmf_type.lower() == "pmfcc":
            nmf = nimfa.Pmf(V, seed="nndsvd", rank=10, max_iter=max_iter,
                            theta=np.random.rand(V.shape[1], V.shape[1]))
            summary = nmf.estimate_rank(rank_range=rank_cands)

        else:
            raise Exception("NMF type is not valid.")

        self.nmf_cv_results = summary
        self.rank_cands = rank_cands

        return summary, nmf

    def plot_cv_resultsV3(self, save_fig_add=False):

        # loading data
        rank_cands = self.rank_cands
        summary = self.nmf_cv_results

        # extracting data
        rss = [summary[rank]['rss'] for rank in rank_cands]
        coph = [summary[rank]['cophenetic'] for rank in rank_cands]
        disp = [summary[rank]['dispersion'] for rank in rank_cands]
        spar = [summary[rank]['sparseness'] for rank in rank_cands]
        spar_w, spar_h = zip(*spar)
        evar = [summary[rank]['evar'] for rank in rank_cands]

        #plt.plot(rank_cands, rss, 'o-', label='RSS', linewidth=2)
        plt.plot(rank_cands, coph, 'o-',
                 label='Cophenetic correlation', linewidth=2)
        plt.plot(rank_cands, disp, 'o-', label='Dispersion', linewidth=2)
        plt.plot(rank_cands, spar_w, 'o-',
                 label='Sparsity (Basis)', linewidth=2)
        plt.plot(rank_cands, spar_h, 'o-',
                 label='Sparsity (Mixture)', linewidth=2)
        plt.plot(rank_cands, evar, 'o-',
                 label='Explained variance', linewidth=2)
        plt.legend(bbox_to_anchor=(0.5, -0.05), ncol=3, numpoints=1)
        plt.show()

        if save_fig_add:
            plt.savefig(fname=save_fig_add,
                        dpi=600, quality=100, format='pdf')

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
        results = Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch=n_jobs)(delayed(self.estimate_componentsV2_helper)(X_train, sparse_mat, n_cmp, n_rep,
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
            nmf_cv_results = Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch=n_jobs,
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
            fig.savefig(fname=save_fig_add,
                        dpi=600,  format='pdf')


# NMF Post processing
def image_threshold(img, disk_size=2, threshold=0.5):
    """
    it is morphology based image thresholding
    using tophat method the function removes small objects in grayscale image
    INPUTs
    image: input image 2D - grayscale without extra channels
    disk_size: size of disk for tophat morphology
    threshold: threshold number (float) to return binary image
    """
    footprint = morphology.disk(disk_size)
    res = morphology.white_tophat(img, footprint)
    remain = img - res

    return remain > threshold


def blob_labeling(image):
    """
    this function calculates properties of detected regions
    sub-functions:
    area
    centroid
    bounding-box
    coordinatios of pixels
    """
    labels_ = measure.label(image, background=0)

    df = pd.DataFrame(measure.regionprops_table(labels_, properties=('centroid',
                                                                     'orientation', 'area', 'bbox', 'coords')))

    df = df.rename(columns={"centroid-0": "center-y(rows)", "centroid-1": "center-x(cols)",
                            "bbox-0": "min_row", "bbox-1": "min_col", "bbox-2": "max_row", "bbox-3": "max_col"})
    return df

# controlling nmf and ROIs


def plot_nmf_ROIs(components, ROIs, base_w_size=4, c_min=0, c_max=3):
    # prepare subplots
    fig, axes = plt.subplots(nrows=components.shape[0], ncols=2,
                             sharex=True, sharey=True,
                             figsize=(base_w_size, int(base_w_size * components.shape[0]/2)), gridspec_kw={'hspace': 0})

    # start plotting
    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            ax.imshow(components[int(i/2)], vmin=c_min, vmax=c_max)
            ax.set_title(f"component # {int(i/2)}")
        elif i % 2 == 1:
            ax.imshow(ROIs[int(i/2)], vmin=c_min, vmax=c_max)
            ax.set_title(f"ROI # {int(i/2)}")

    plt.tight_layout()
    plt.show()


# connectivity analysis
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
    def __init__(self, data, labels=False):
        self.data = data
        self.n = data.shape[0]
        self.nr_row = data.shape[1]
        self.nr_col = data.shape[2]
        self.labels = labels

    # apply preprocessing step - spatially subsample data and store location and values in list
    def preprocessing(self, rows_start_end, cols_start_end, resolution=5):
        # inform user about symmetry of selected dots between hemispheres
        print("*********************************************************************************")
        print("***** selected dots have to be symmetric between left and right hemispheres *****")
        print("*********************************************************************************")

        # get tensor data
        self.resolution = resolution

        # initialize location value list
        loc_val = []

        # get location and traces (when plotting notice that col is x and row is y)
        for i in range(cols_start_end[0], cols_start_end[1], resolution):
            for j in range(rows_start_end[0], rows_start_end[1], resolution):
                loc_val.append([j, i, self.data[:, j, i]])

        self.location_value = loc_val

        # if labels are not provided
        if not self.labels:
            self.labels = self.show_locations()

        return loc_val

    # if labels are not provided get labels per location
    def show_locations(self):

        # reading locations from preprocessing function and asking user for label information
        loc_val = self.location_value

        # getting data and prepare average frame
        img = self.np.mean(self.data, axis=0)

        # initialize figure
        fig, ax = self.plt.subplots(1, 1, figsize=(10, 10))

        # plot img and add seleted points withred circles
        ax.imshow(img)
        for i, item in enumerate(loc_val):
            ax.plot(item[1], item[0], 'ro', ms=10)
            ax.annotate(str(i), xy=(item[1], item[0]))

    def show_labels(self, labels, fig_size=(10, 10), font_size=5):
        # annotating labels on image
        loc_val = self.location_value

        # getting data and prepare average frame
        img = self.np.mean(self.data, axis=0)

        # initialize figure
        fig, ax = self.plt.subplots(1, 1, figsize=fig_size)

        # plot img and add seleted points withred circles
        ax.imshow(img)
        for i, item in enumerate(loc_val):
            ax.plot(item[1], item[0], 'ro', ms=10)
            ax.annotate(labels[i], xy=(item[1], item[0]), fontsize=font_size)

    # estimate sparse covariance matrix
    def covariance_lasso(self, location_value, alpha=10, max_iter=200, mode='cd', n_jobs=-1):

        # get location value from preprocessing
        loc_val = location_value

        # prepare data
        data = [item[2] for item in loc_val]
        data = self.np.stack(data)
        data = data.T

        # fitting model
        cov = GraphicalLassoCV(
            alphas=alpha, max_iter=max_iter, mode=mode, n_jobs=n_jobs).fit(data)

        self.model = cov
        return cov

    # plotting sparsity alpha value
    def plot_model_selection(self):
        model = self.model
        plt.figure(figsize=(6, 4))

        plt.plot(model.cv_alphas_,
                 np.mean(model.grid_scores_, axis=1), 'o-', label='tested alphas')
        plt.axvline(model.alpha_, color='.5',
                    label=f'optimal alpha = {np.round(model.alpha_, 2)}')
        plt.legend()
        plt.title('Model selection')
        plt.ylabel('Cross-validation score')
        plt.xlabel('alpha')

        plt.show()

    # plot graph
    def plot_network_graph(self, adj_matrix, labels, config, title, save_add=False, font_size=10, fig_size=(10, 10), nr_lines=300):
        """
        INPUTs
        adj_matrix: n*n connectivity matrix. False means covariance_lasso is calculated 
        labels: label name corresponding to each individual i row/col in conenctivity matrix
        config: configuration dictionary indicating color code information to given brain regions
        cov_prec: if using covariance_lasso, plot convaraince or prescision. If True cov and if False recision
        """

        # solving labels duplicate problem(it is necessary for layout)
        all_labels = self.check_label_duplicate(labels)

        # sorting nodes to get circular right-left hemispher
        index_sorted, node_colors = self. sort_labels(
            all_labels=all_labels, config=config)

        # define circular layout and get node angles
        node_angles = circular_layout(all_labels, [all_labels[i] for i in index_sorted], start_pos=90,
                                      group_boundaries=[0, len(all_labels) / 2])

        # plot connectivity map
        fig = plt.figure(num=None, figsize=fig_size, facecolor='black')
        fig, ax = plot_connectivity_circle(adj_matrix, all_labels, node_colors=node_colors,
                                           node_angles=node_angles,
                                           title=title, fontsize_names=font_size, fig=fig, n_lines=nr_lines)

        if save_add:
            fig.savefig(fname=save_add,
                        dpi=600, format='pdf')

    # solve duplicates issue automatically

    def check_label_duplicate(self, labels):
        # check if duplicates happen in the label list

        # initialize lists
        my_list = []
        out = []

        # check duplicates in for loop and if happens correct them
        for label in labels:
            if label in my_list:

                # get repeatition of duplicate
                # find where _ is happening
                under_line = label.find('_')

                # find name of region
                key_word = label[:under_line]

                # search in output list for all words starting with key_word
                all_dup = []
                all_dup = [lb for lb in out if lb.startswith(key_word)]

                # always take the last one
                out.append(self.label_helper(my_string=all_dup[-1]))
            else:
                out.append(label)
                my_list.append(label)
        return out

    def label_helper(self, my_string):
        # this is a helper function to add digit on duplicate

        # first find where _ is
        under_line = my_string.find('_')

        # check if character before _ is numberic or not and if not add number if yes add 1 on number
        if my_string[under_line-2:under_line].isnumeric():
            my_string = my_string.replace(
                my_string[under_line-2:under_line], str(int(my_string[under_line-2:under_line]) + 1))
        else:
            my_string = my_string[:under_line] + \
                str(10) + my_string[under_line:]

        return my_string

    def sort_labels(self, all_labels, config):
        all_labels = self.check_label_duplicate(all_labels)

        index_rh = []
        index_lh = []

        for k, v in config.items():
            index_rh = index_rh + \
                [i for i, ll in enumerate(all_labels) if (
                    ll.startswith(k) and ll.endswith('right'))]
            index_lh = index_lh + \
                [i for i, ll in enumerate(all_labels) if (
                    ll.startswith(k) and ll.endswith('left'))]

        # starting with indexlh because start_pos of cicular_layout start at 90 degree
        # and because we continue in anti-clockwise direction for right hemisphere I have to flip indices
        index_sorted = index_lh + index_rh[::-1]

        # create colors
        my_colors = []
        for name in all_labels:
            for name2 in config.keys():
                if name.startswith(name2):
                    my_colors.append(config[name2][0])

        return index_sorted, my_colors


def video_player(np_array_video, cmin=0, cmax=1, intervals_=50):

    # np array with shape (frames, height, width, channels)
    if len(np_array_video.shape) == 3:
        np_array_video = np_array_video[..., np.newaxis]

    video = np.array(np_array_video)

    fig = plt.figure()
    im = plt.imshow(video[0, :, :, :], vmin=cmin, vmax=cmax)

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


""" NOT using it at the moment
# https://www.python-graph-gallery.com/406-chord-diagram_mne
def plot_network(adj, labels, weight_norm = 1):
    
    plotting circular graph from given adjancy matrix and label names
   
    # import library
    import networkx as nx
    
    # create graph
    g = nx.from_numpy_matrix(adj)
    
    # initialize label dict and add label names
    label_dict = {}
    for i in range(12):
        label_dict.update({i:labels[i]})

    # get full edge information
    edges = g.edges()

    # prepare figure, calculate weights and set draw options
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (6,6))
    weights = [g[u][v]['weight']/weight_norm for u,v in edges]
    options = {
        "node_color": "#A0CBE2",
        "edge_color": weights,
        "labels": label_dict,
        "width":weights,
        "edge_cmap": plt.cm.autumn,
        "with_labels": True,
        "node_size":300
    }
    # plot graph
    nx.draw(g, pos = nx.circular_layout(g), ax = ax, **options)




    adjc = np.random.randint(low = 0, high = 40, size =(12,12))
lb = ['a', 'b', 'c', 'd', 'e','f', 'h', 'g', 'l', 'm', 'n','o']
plot_network(adj=adjc, 
             labels=lb, weight_norm = 7)
"""
