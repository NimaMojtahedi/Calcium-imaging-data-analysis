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

    return rank, replicates, tr_error, te_error


# filtering lowpass, high pass, bandpass, ...
