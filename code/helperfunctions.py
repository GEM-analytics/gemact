"""
This script contains helper function to be used in the main scripts "lossmodel.py","lossreserve.py" and "lossaggregation.py".
"""

import numpy as np
from scipy.interpolate import interp1d
import sobol as sbl

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('hfns')

def ecdf(x_):
    """
    It computes the empirical cumulative density function.

    Empirical cumulative density function computed on the vector x_.

    Parameters:
    x_ (numpy.ndarray): sequence of values to compute the ecdf on.

    Returns:
    x_(numpy.ndarray): starting sequence.
    f(x_)(numpy.ndarray): empirical cumulative density function.
    """
    dim = len(x_)
    x_ = np.sort(x_)
    y_ = np.cumsum(np.repeat(1, dim)) / dim
    f = interp1d(x_, y_)

    return x_, f(x_)

def sobol_generator(n, dim,skip=0):
    """
    Wrapper to generate sobol sequence.

    It generates a dim-dimensional sobol sequence from the script sobol.py.

    Parameters:
    n (int): length of the sobol sequence.
    dim (int): dimension of the sobol sequence.

    Returns:
    numpy.ndarray: generated sobol sequence.

    """
    return sbl.i4_sobol_generate(m=dim, n=n,skip=skip)

#LossReserve
def normalizerNans(x_):
    """
    It normalizes a vector with nan values.

    It normalizes a vector with nan values ignoring the nan values during the computation.

    Parameters:
    x_ (numpy.ndarray): sequence to be normalized.

    Returns:
    numpy.ndarray: normalized sequence.
    """
    if np.sum(np.isnan(x_)) < x_.shape[0]:
        x_[~np.isnan(x_)]=x_[~np.isnan(x_)]/np.sum(x_[~np.isnan(x_)])
    return x_

class ClaytonCDF:
    """
    Frank cumulative distribution function.

    :param par: Frank copula parameter.
    :type par: float

    """
    def __init__(self,par):
        self.par =par
    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, var):
        if var < 0:
            raise ValueError("The input 'par' must be non-negative")
        self.__par = var

    def cdf(self,x):
        """
        :param x: Array with shape (N,d) where N is the number of points and d the dimension
        :type x: numpy.ndarray

        :return: Frank c.d.f. for each row of x.
        :rtype: numpy.ndarray

        """
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                return (np.sum(np.minimum(x, 1) ** (-self.par) - 1) + 1) ** -(1 / self.par)
        N = len(x)
        output = np.array([0.] * N)
        PosIndex = ~np.array(x <= 0).any(axis=1)
        output[PosIndex] = (np.sum(np.minimum(x[PosIndex, :], 1) ** (-self.par) - 1, axis=1) + 1) ** (-1 / self.par)
        return output

##### Frank #####
class FrankCDF:
    """
    Frank cumulative distribution function.

    :param par: Frank copula parameter.
    :type par: float

    """

    def __init__(self, par):
        self.par = par

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, var):
        if var < 0:
            raise ValueError("The input 'par' must be non-negative")
        self.__par = var

    def cdf(self, x):
        """
        :param x: Array with shape (N,d) where N is the number of points and d the dimension
        :type x: numpy.ndarray

        :return: Frank c.d.f. for each row of x.
        :rtype: numpy.ndarray

        """
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                d = len(x)
                return -1 / self.par * np.log(
                    1 + np.prod(np.exp(-self.par * np.minimum(x, 1)) - 1) / (np.exp(-self.par) - 1) ** (d - 1))
        N = len(x)
        d = len(x[0])
        output = np.array([0.] * N)
        PosIndex = ~np.array(x <= 0).any(axis=1)
        output[PosIndex] = -1 / self.par * np.log(
            1 + np.prod(np.exp(-self.par * np.minimum(x[PosIndex, :], 1)) - 1, axis=1) / (np.exp(-self.par) - 1) ** (d - 1))
        return output

## GumbelCDF
class GumbelCDF:
    """
    Gumbel cumulative distribution function.

    :param par: Gumbel copula parameter.
    :type par: float

    """

    def __init__(self, par):
        self.par = par

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, var):
        if var < 0:
            raise ValueError("The input 'par' must be non-negative")
        self.__par = var

    def cdf(self, x):
        """
        :param x: Array with shape (N,d) where N is the number of points and d the dimension
        :type x: numpy.ndarray

        :return: Gumbel c.d.f. for each row of x.
        :rtype: numpy.ndarray

        """
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                return np.exp(-np.sum((-np.log(np.minimum(x, 1))) ** self.par) ** (1 / self.par))
        N = len(x)
        output = np.array([0.] * N)
        PosIndex = ~np.array(x <= 0).any(axis=1)
        output[PosIndex] = np.exp(-np.sum((-np.log(np.minimum(x[PosIndex, :], 1))) ** self.par, axis=1) ** (1 / self.par))
        return output

def lrcrm_f1(x,dist):
    """
    It simulates a random number from a poisson distribution.
    It simulates a random number from a distribution a poisson distribution with parameter mu.

    :param x: distribution parameter.
    :type x: float
    :param dist: poisson distribution.
    :type dist: scipy.stats._discrete_distns.poisson_gen

    :return:simulated random number.
    :rtype: numpy.ndarray
    """
    return dist(mu=x).rvs(1)

def lrcrm_f2(x,dist):
    """
    It simulates random values from a gamma.

    Parameters:
    :param x: it contains the gamma parameters and the number of random values to be simulated.
    :type x: numpy.ndarray
    :param dist: gamma distribution.
    :type dist: scipy.stats._discrete_distns.gamma_gen

    :return: sum of the simulated numbers.
    :rtype: numpy.ndarray
    """
    return np.sum(dist(a=x[1],scale=x[2]).rvs(int(x[0])))

def cartesian_product(*arrays):
    """
    Generates the matrix points where copula is computed.

    :param d: dimension.
    :type d: ``int``
    :return: matrix of points.
    :rtype:``numpy.ndarray``
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows)