from .libraries import *
from . import helperfunctions as hf

quick_setup()
logger = log.name('copulas')

# Coupla interface
class ICopula:
    """
    Copula informal interface.
    """
    pass

# Clayton
class ClaytonCopula(ICopula):
    """
    Clayton copula.

    :param par: copula parameter.
    :type par: ``float``
    :param dim: copula dimension.
    :type dim: ``int``
    """

    def __init__(self, par, dim):
        self.par = par
        self.dim = dim

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(value, 'par', logger, (float, int, np.floating), lower_bound=0)
        self.__par = value

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, value):
        hf.assert_type_value(value, 'dim', logger, (float, int, np.floating), lower_bound=1)
        value = int(value)
        self.__dim = value

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: Array with shape (N, d) where N is the number of points and d the dimension.
        :type x: ``numpy.ndarray``
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        hf.check_condition(
            x.shape[1], self.dim, 'x', logger
        )
        
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                return (np.sum(np.minimum(x, 1) ** (-self.par) - 1) + 1) ** -(1 / self.par)

        capital_n = len(x)
        output = np.array([0.] * capital_n)
        index = ~np.array(x <= 0).any(axis=1)
        output[index] = (np.sum(np.minimum(x[index, :], 1) ** (-self.par) - 1, axis=1) + 1) ** (-1 / self.par)
        return output

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        gamma_sim = stats.gamma.rvs(1 / self.par, size=[size, 1], random_state=random_state)
        exp_sim = stats.gamma.rvs(1, size=[size, self.dim], random_state=random_state + 2)
        output = (1 + exp_sim / gamma_sim) ** (-1 / self.par)
        return output


# Frank
class FrankCopula(ICopula):
    """
    Frank copula.

    :param par: copula parameter.
    :type par: ``float``
    :param dim: copula dimension.
    :type dim: ``int``
    """

    def __init__(self, par, dim):
        self.par = par
        self.dim = dim

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(
            value, 'par', logger, (int, float), lower_bound=0, lower_close=False
            )
        self.__par = value

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, value):
        hf.assert_type_value(value, 'dim', logger, (float, int, np.floating), lower_bound=1)
        value = int(value)
        self.__dim = value

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: Array with shape (N, d) where N is the number of points and d the dimension.
        :type x: ``numpy.ndarray``
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        hf.check_condition(
            x.shape[1], self.dim, 'x', logger
        )
        
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                d = len(x)
                return -1 / self.par * np.log(
                    1 + np.prod(np.exp(-self.par * np.minimum(x, 1)) - 1) / (np.exp(-self.par) - 1) ** (d - 1))
        capital_n = len(x)
        d = len(x[0])
        output = np.array([0.] * capital_n)
        index = ~np.array(x <= 0).any(axis=1)
        output[index] = -1 / self.par * np.log(
            1 + np.prod(np.exp(-self.par * np.minimum(x[index, :], 1)) - 1, axis=1) / (
                    np.exp(-self.par) - 1) ** (d - 1))
        return output

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        logarithmic_sim = stats.logser.rvs(1 - np.exp(-self.par), size=[size, 1], random_state=random_state)
        exp_sim = stats.gamma.rvs(1, size=[size, self.dim], random_state=random_state)
        output = -1 / self.par * np.log(1 + np.exp(-exp_sim / logarithmic_sim) * (np.exp(-self.par) - 1))
        return output


# Gumbel
class GumbelCopula(ICopula):
    """
    Gumbel copula.

    :param par: copula parameter.
    :type par: ``float``
    :param dim: copula dimension.
    :type dim: ``int``
    """

    def __init__(self, par, dim):
        self.par = par
        self.dim = dim

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, value):
        hf.assert_type_value(value, 'par', logger, (float, int, np.floating), lower_bound=0)
        self.__par = value

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, value):
        hf.assert_type_value(value, 'dim', logger, (float, int, np.floating), lower_bound=1)
        value = int(value)
        self.__dim = value

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: Array with shape (N, d) where N is the number of points and d the dimension.
        :type x: ``numpy.ndarray``
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        hf.check_condition(
            x.shape[1], self.dim, 'x', logger
        )
        
        if len(x.shape) == 1:
            if (x <= 0).any():
                return 0
            else:
                return np.exp(-np.sum((-np.log(np.minimum(x, 1))) ** self.par) ** (1 / self.par))

        output = np.array([0.] * x.shape[0])
        index = ~np.array(x <= 0).any(axis=1)
        output[index] = np.exp(-np.sum((-np.log(np.minimum(x[index, :], 1))) ** self.par, axis=1) ** (1 / self.par))
        return output

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        a_ = 1 / self.par
        uniform_sim = (stats.uniform.rvs(size=[size, 1], random_state=random_state) - 0.5) * np.pi
        exp_sim = stats.gamma.rvs(1, size=[size, 1], random_state=random_state)
        stable_sim = np.sin(a_ * (np.pi / 2 + uniform_sim)) / np.cos(
                    uniform_sim) ** (1 / a_) * (
                                 np.cos(uniform_sim - a_ * (np.pi / 2 + uniform_sim)) / exp_sim) ** ((1 - a_) / a_)
        exp_sim = stats.gamma.rvs(1, size=[size, self.dim], random_state=random_state + 2)
        output = np.exp(-(exp_sim / stable_sim) ** (1 / self.par))
        return output


# Gaussian
class GaussCopula(ICopula):
    """
    Gaussian copula.

    :param corr: Correlation matrix.
    :type corr: ``numpy.ndarray``
    """

    def __init__(self, corr):
        self.corr = corr

    @property
    def corr(self):
        return self.__corr

    @corr.setter
    def corr(self, value):
        hf.assert_type_value(value, 'corr', logger, (np.ndarray))
        if not np.allclose(value, np.transpose(value)):
            raise ValueError('corr must be a symmetric square matrix')
        if not np.allclose(np.diagonal(value), np.ones(value.shape[0])):
            raise ValueError('%r is not a correlation matrix' % value)
        self.__corr = value

    @property
    def dim(self):
        return self.corr.shape[0]

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: Array with shape (N, d) where N is the number of points and d the dimension.
        :type x: ``numpy.ndarray``
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        return stats.multivariate_normal.cdf(stats.norm.ppf(x), cov=self.corr)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        sim = stats.multivariate_normal.rvs(
            mean=np.zeros(self.dim),
            cov=self.corr,
            size=size,
            random_state=random_state
        )
        return stats.norm.cdf(sim)


# TCopula
class TCopula(ICopula):
    """
    T-Student copula.

    :param corr: Correlation matrix.
    :type corr: ``numpy.ndarray``
    :param df: Degree of freedom.
    :type df: ``int``
    """

    def __init__(self, corr, df):
        self.corr = corr
        self.df = df
        self.__error_cdf = None

    @property
    def dim(self):
        return self.corr.shape[0]

    @property
    def corr(self):
        return self.__corr

    @corr.setter
    def corr(self, value):
        hf.assert_type_value(value, 'corr', logger, (np.ndarray))
        if not np.allclose(value, np.transpose(value)):
            raise ValueError("corr must be a symmetric square matrix")
        if not np.allclose(np.diagonal(value), np.ones(value.shape[0])):
            raise ValueError('%r is not a correlation matrix' % value)
        self.__corr = value

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, value):
        hf.assert_type_value(value, 'df', logger, (int, float), lower_bound=1)
        self.__df = value

    @property
    def error_cdf(self):
        return self.__error_cdf

    def cdf(self, x, tolerance=1e-4, n_iterations=30):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
                    Array with shape (n, d) where n is the number of data points and d the dimension.
        :type x: ``numpy.ndarray``
        :param tolerance: tolerance threshold of approximation (default is 1e-4).
        :type tolerance: ``float``, optional
        :param n_iterations: number of iteration (default is 30).
        :type n_iterations: ``int``, optional
        
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        q = stats.t(self.df).ppf(x)
        (prob, err) = hf.multivariate_t_cdf(q, self.corr, self.df, tolerance, n_iterations)
        self.__error_cdf = err
        return prob

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        sim = stats.multivariate_t.rvs(
            df=self.df,
            shape=self.corr,
            size=size,
            random_state=random_state
        )
        return stats.t.cdf(sim, df=self.df)


# Independence
class IndependenceCopula(ICopula):
    """
    The product (independence) copula.

    :param dim: copula dimension.
    :type dim: ``int``
    """

    def __init__(self, dim):
        self.dim = dim

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, value):
        hf.assert_type_value(value, 'dim', logger, (float, int, np.floating), lower_bound=1)
        value = int(value)
        self.__dim = value

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: Array with shape (N, d) where N is the number of points and d the dimension.
        :type x: ``numpy.ndarray``
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        try:
            x = x.reshape(-1, self.dim)
        except Exception:
            logger.error('Please make sure x dimension is the same as copula dimension')
            raise

        return np.prod(x, axis=1)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        return np.random.uniform(size=(size, self.dim))


# Fréchet–Hoeffding Lower Bound 
class FHLowerCopula(ICopula):
    """
    Fréchet–Hoeffding lower bound bidimensional copula.
    """

    def __init__(self):
        pass

    @staticmethod
    def dim():
        return 2

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: Array with shape (N, 2) where N is the number of points.
        :type x: ``numpy.ndarray``
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        try:
            x = x.reshape(-1, 2)
            logger.warning('x shape second dimension set to 2')
        except Exception:
            logger.error('Please make sure x shape second dimension is 2')
            raise

        return np.maximum(np.sum(x, axis=1) - 1, 0)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        u = np.random.uniform(size=(size, 1))
        return np.concatenate((u, 1-u), axis=1)


# Fréchet–Hoeffding Upper Bound 
class FHUpperCopula(ICopula):
    """
    Fréchet–Hoeffding upper bound copula.
    """

    def __init__(self, dim):
        self.dim = dim

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, value):
        hf.assert_type_value(value, 'dim', logger, (float, int, np.floating), lower_bound=1)
        value = int(value)
        self.__dim = value

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: Array with shape (N, d) where N is the number of points and d the dimension.
        :type x: ``numpy.ndarray``
        :return: Cumulative distribution function in x.
        :rtype: ``numpy.ndarray``
        """
        try:
            x = x.reshape(-1, self.dim)
        except Exception:
            logger.error('Please make sure x dimension is the same as copula dimension')
            raise

        return np.min(x, axis=1)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        u = np.random.uniform(size=(size, 1))
        return np.tile(u, (1, self.dim))
    