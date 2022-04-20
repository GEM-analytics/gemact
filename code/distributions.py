import numpy as np
# from scipy import rand ---> remove?
import scipy.stats
import scipy.special
# from datetime import datetime ---> remove?
# from sqlalchemy import null ---> remove?
from twiggy import quick_setup, log

quick_setup()
logger= log.name('lossaggregation')

## Distribution Wrapper Class
class _DiscrDistributionWrapper():

    """
    Wrapper for discrete distributions. Parent (private) class to be inherited.
    """

    def __init__(self):
        self.__dist = None

    @property
    def dist(self):
        return self.__dist

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default=1).
        :type size: int, optional
        :param random_state: random state for the random number generator (default=None).
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        random_state = 1234 if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')

        return self.dist.rvs(size=size, random_state=random_state)

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.pmf(k=k)

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int
        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.logpmf(k=k)

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.cdf(k=k)

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.logcdf(k=k)

    def sf(self, k):
        """
        Survival function.

        :param k: quantile where the survival function is evaluated.
        :type k: int
        :return: survival function
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.sf(k=k)

    def logsf(self, k):
        """
        Natural logarithm of the survival function.

        :param k: quantile where the logarithm of the survival function is evaluated.
        :type k: int
        :return: natural logarithm of the survival function
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.logsf(k=k)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: level at which the inverse survival function is evaluated.
        :type q: float
        :return: inverse survival function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.dist.ppf(1 - q)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’) function.

        :param moments: moments to be returned.
        :type moments: string, optional
        :return: moments.
        :rtype: tuple

        """
        return self.dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the rv.

        :return: entropy
        :rtype: numpy.ndarray

        """
        return self.dist.entropy()

    def expect(self, func, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: function for which integral is calculated. Takes only one argument. The default is the identity mapping f(x) = x.
        :type func: callable, optional
        :param lb: Lower bound for integration. Default is set to the support of the distribution.
        :type lb: scalar, optional
        :param ub: Upper bound for integration. Default is set to the support of the distribution.
        :type ub: scalar, optional
        :param conditional: If True, the integral is corrected by the conditional probability of the integration interval.
        The return value is the expectation of the function, conditional on being in the given interval. Default is False.
        :type conditional: bool, optional
        :return: the calculated expected value.
        :rtype: float

        """
        return self.dist.expect(func, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median.
        :rtype: float

        """
        return self.dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: float

        """
        return self.dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: float

        """
        return self.dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: float

        """
        return self.dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :type alpha: float
        :return: Endpoints
        :rtype: tuple

        """
        return self.dist.interval(alpha=alpha)

## Poisson Distribution Class
class Poisson(_DiscrDistributionWrapper):
    """
    Poisson distribution. Wrapper to scipy poisson distribution.
    Refer to :py:class:'~_DiscrDistributionWrapper' (its parent class) for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: int, optional
    :param dist: (private) scipy reference distribution.
    :type dist: ``scipy.stats._discrete_distns.poisson_gen``

    # to remove ---
    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) rv probability in zero.
    :type p0: float

    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Poisson distribution parameter mu (rate).

    """

    name = 'poisson'
    def __init__(self, loc=0, **kwargs):
        super().__init__()
        self.__mu = kwargs['mu']
        self.__loc = loc
        self.__dist = scipy.stats.poisson(mu=self.mu, loc=self.loc)
        self.__a = 0
        self.__b = self.mu
        self.__p0 = np.exp(-self.mu)

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert (value > 0), logger.error("mu has to be > 0")
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b

    @property
    def p0(self):
        return self.__p0

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f: punto in cui viene valutata la funzione...
        :type f: numpy array (supporto distr)
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """
        return np.exp(self.b * (f - 1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: numpy.array
        """
        return self.a, self.b, self.p0

## Binomial Distribution Class
class Binom(_DiscrDistributionWrapper):
    """
    Binomial distribution. Wrapper to scipy binomial distribution.
    Refer to :py:class:'~_DiscrDistributionWrapper' (its parent class) for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: int, optional
    :param dist: (private) scipy reference distribution.
    :type dist: ``scipy.stats._discrete_distns.binom_gen``
    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) rv probability in zero.
    :type p0: float
    
    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *n* (``int``) --
          Number of trials.
        * *p* (``float``) --
          Probability of a success, parameter of the binomial distribution.

    """
    name = 'binom'
    def __init__(self, loc=0, **kwargs):
        self.__n = kwargs['n']
        self.__p = kwargs['p']
        self.__loc = loc
        self.__dist = scipy.stats.binom(n=self.n, p=self.p, loc=self.loc)
        self.__a = -self.p / (1 - self.p)
        self.__b = (self.n + 1) * (self.p/(1 - self.p))
        self.__p0 = (1 - self.p) ** self.n
    
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and (value > 0)),\
             logger.error("n has to be a positive integer")
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error("p has to be in [0, 1]")
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """
        return (1 + self.a / (self.a - 1)*(f - 1))**(-self.b/self.a-1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: int
        """
        return self.a, self.b, self.p0

## Geometric Distribution Class
class Geom(_DiscrDistributionWrapper):
    """
    Geometric distribution. Wrapper to scipy geometric distribution.
    Refer to :py:class:'~_DiscrDistributionWrapper' (its parent class) for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: int, optional
    :param dist: (private) scipy reference distribution.
    :type dist: ``scipy.stats._discrete_distns.geom_gen``
    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) rv probability in zero.
    :type p0: float

    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *p* (``float``) --
          Probability parameter of the geometric distribution.

    """

    name = 'geom'
    def __init__(self, loc=0, **kwargs):
        self.__p = kwargs['p']
        self.__loc = loc
        self.__dist = scipy.stats.geom(p=self.p, loc=self.loc)
        self.__a = 1 - self.p
        self.__b = 0
        self.__p0 = np.array([((1 - self.p)/self.p)**(-1)])

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error("p has to be in [0, 1]")
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """
        return (1- self.a/(1-self.a) * (f-1))**(-1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: int
        """
        return self.a, self.b, self.p0

## Negative Binomial Class
class NegBinom(_DiscrDistributionWrapper):
    """
    Negative Binomial distribution. Wrapper to scipy negative binomial distribution.
    Refer to :py:class:'~_DiscrDistributionWrapper' (its parent class) for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: int, optional
    :param dist: (private) scipy reference distribution.
    :type dist: ``scipy.stats._discrete_distns.nbinom_gen``
    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) rv probability in zero.
    :type p0: float

    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *n* (``int``) --
          Size parameter of the negative binomial distribution.
        * *p* (``float``) --
          Probability parameter of the negative binomial distribution.

    """

    name = 'nbinom'
    def __init__(self, loc=0, **kwargs):
        self.__n = kwargs['n']
        self.__p = kwargs['p']
        self.__loc = loc
        self.__dist = scipy.stats.nbinom(n=self.n, p=self.p, loc=self.loc)
        self.__a = 1 - self.p
        self.__b = (self.n - 1)*(1 - self.p)
        self.__p0 = np.array([self.p**self.n])
    
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and (value > 0)),\
             logger.error("n has to be a positive integer")
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1,\
            logger.error('p must be between zero and one.')
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """
        return (1- self.a/(1-self.a) * (f-1))**(-self.b/self.a - 1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: int
        """
        return self.a, self.b, self.p0

## Zero-truncated Poisson Distribution Class
class ZTPoisson:
    """
    Zero-truncated Poisson distribution. Poisson distribution with no mass (truncated) in 0.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: int, optional
    :param dist: (private) scipy reference non-zero-truncated distribution.
    :type dist: ``scipy.stats._discrete_distns.poisson_gen``
    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) base rv probability in zero.
    :type p0: float

    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Zero-truncated Poisson distribution parameter mu (rate).

    """

    name = 'ztpoisson'
    def __init__(self, loc=0, **kwargs):
        self.__mu = kwargs['mu']
        self.__loc = loc
        self.__dist = scipy.stats.poisson(mu=self.mu, loc=self.loc)
        self.__a = 0
        self.__b = self.mu
        self.__p0 = np.exp(-self.mu)
        
    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert value > 0, logger.error('mu has to be positive')
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """

        temp = self.dist.pmf(k)/(1 - self.p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.0
            else:
                temp[zeros] = 0.0
            return temp
    
    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """

        return (self.dist.cdf(k) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """

        random_state = 1234 if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')
        
        np.random.seed(random_state)
        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """

        p_ = np.array(q)
        return (self.dist.ppf(q=p_*(1 - self.dist.cdf(0)) + self.dist.cdf(0)))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray

        """

        return (np.exp(self.b*f) - 1)/(np.exp(self.b) - 1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: int

        """

        return self.a, self.b, 0

## Zero-modified Poisson
class ZMPoisson:
    """
    Zero-modified Poisson distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified Poisson distribution.

    :param loc: location parameter (default=0).
    :type loc: ``float``, optional 
    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: ``float``
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: ``float``
    :param p0: (private) base Poisson distribution probability mass in zero.
    :type p0: ``float``
    :param dist: (private) scipy reference non-zero-modified distribution.
    :type dist: ``scipy.stats._discrete_distns.poisson_gen``
    :param maxDiff: (private) threshold to determine which method to generate random variates.
    :type maxDiff: ``float`` (fixed)

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Zero-modified Poisson distribution rate parameter.
        * *p0M* (``numpy.float64``) --
          Zero-modified Poisson mixing parameter. Resulting probability mass in zero.

    """

    name = 'ZMpoisson'
    def __init__(self, loc=0, **kwargs):
        self.__loc = loc
        self.__mu = kwargs['mu']
        self.__p0M = kwargs['p0M']
        self.__dist = scipy.stats.poisson(mu=self.mu, loc=self.loc)
        self.__a = 0
        self.__b = self.mu
        self.__p0 = np.exp(-self.mu)
        self.__maxDiff = 0.95

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        assert isinstance(value, int), logger.error("loc has to be int type")
        self.__loc = value

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        assert value > 0, logger.error('mu must be positive')
        self.__mu = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert (0 <= value <= 1), logger.error('p0M must be between zero and one.')
        self.__p0M = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    @property
    def maxDiff(self):
        return self.__maxDiff

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        
        temp = self.dist.pmf(k)*(1 - self.p0M)/(1 - self.p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = self.p0M
            else:
                temp[zeros] = self.p0M
            return temp

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.p0M + (1 - self.p0M)*(self.dist.cdf(k) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        random_state = 1234 if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        if (self.mu == 0):
            u_ = np.random.uniform(0, 1, size)
            idx = u_ <= self.p0M
            u_[idx] = 0
            u_[np.invert(idx)] = 1
            return u_
        
        if (self.p0M >= self.p0):
            u_ = np.random.uniform(0, (1 - self.p0), size)
            idx = (u_ <= (1 - self.p0M))
            u_[idx] = self.dist.rvs(mu=self.mu, size=np.sum(idx))
            u_[np.invert(idx)] = 0
            return u_
       
        if ((self.p0 - self.p0M) < self.maxDiff):
            # rejection method
            u_ = []
            while len(u_) < size:
                x_ = self.dist.rvs(1, self.mu)
                if (x_ != 0 or np.random.uniform(0, self.p0*(1 - self.p0M), 1) <= (1 - self.p0)*self.p0M):
                    u_.append(x_)
            return np.asarray(u_)
        else:
            # inversion method
            u_ = np.random.uniform((self.p0 - self.p0M)/(1 - self.p0M), 1, size)
            return self.dist.ppf(u_, self.mu)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        
        q_ = np.array(q)
        temp = self.dist.ppf((1 - self.dist.cdf(0))*(q_ - self.p0M)/(1 - self.p0M)+self.dist.cdf(0))
        return temp

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """
        return self.p0M+(1 - self.p0M)*(np.exp(self.b*f) - 1)/(np.exp(self.b) - 1)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, self.p0M 

## Zero-truncated binomial
class ZTBinom:
    """
    Zero-truncated binomial distribution. Binomial distribution with no mass (truncated) in 0.

    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) base rv probability in zero.
    :type p0: float
    :param dist: (private) scipy reference non-zero-truncated distribution.
    :type dist: ``scipy.stats._discrete_distns.binom_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-trincated binomial distribution size parameter n.
        * *p*(``float``) --
          Zero-truncated binomial distribution probability parameter p.

    """
    name = 'ZTbinom'
    def __init__(self, **kwargs):
        self.__n = kwargs['n']
        self.__p = kwargs['p']
        self.__dist = scipy.stats.binom(n=self.n, p=self.p)
        self.__a = -self.p/(1 - self.p)
        self.__b = (self.n + 1)*(self.p/(1 - self.p))
        self.__p0 = np.array([(1 - self.p)**self.n])

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and value >= 1), logger.error('n must be a natural number')
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error('p must be between zero and one.')
        self.__p = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        
        temp = self.dist.pmf(k)/(1 - self.p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return (self.dist.cdf(k) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        random_state = 1234 if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """

        q_ = np.array(q)
        return self.dist.ppf(q=q_ * (1 - self.dist.cdf(0)) + self.dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """
        a_ = self.a
        b_ = self.b
        return ((1+a_/(a_ - 1)*(f - 1))**(-b_/a_-1)-(1 - a_)**(b_/a_+1))/(1-(1 - a_)**(b_/a_+1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, 0

## Zero-modified binomial
class ZMBinom:
    """
    Zero-modified binomial distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified binomial distribution.

    :param dist: (private) scipy reference non-zero-modified distribution.
    :type dist: ``scipy.stats._discrete_distns.binom_gen``
    :param distZT: (private) reference zero-truncated distribution.
    :type distZT: :py:class:'~ZTBinom'
    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) base rv probability in zero.
    :type p0: float

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``numpy.float64``) --
          Zero-modified binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-modified binomial distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          Zero-modified binomial mixing parameter.
          
    """

    name = 'ZMbinom'
    def __init__(self, **kwargs):
        self.__n = kwargs['n']
        self.__p = kwargs['p']
        self.__p0M = kwargs['p0M']
        self.__dist = scipy.stats.binom(n=self.n, p=self.p)
        self.__distZT= ZTBinom(n=self.n, p=self.p)
        self.__a = -self.p / (1 - self.p)
        self.__b = (self.n + 1) * (self.p / (1 - self.p))
        self.__p0 = np.array([(1 - self.p) ** self.n])

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert (isinstance(value, int) and value >= 1), logger.error('n must be a natural number')
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert (0 <= value <= 1), logger.error('p must be between zero and one.')
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert (0 <= value <= 1), logger.error('p0M must be between zero and one.')
        self.__p0M = value

    @property
    def dist(self):
        return self.__dist

    @property
    def distZT(self):
        return self.__distZT

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    def pmf(self, k):
        """
        Probability mass function

        :param k: probability mass function will be computed in k.
        :type k: int

        :return: pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.dist.pmf(k)*(1 - self.p0M)/(1 - self.p0)

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.p0M + (1 - self.p0M)*(self.dist.cdf(k) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
	    
        random_state = 1234 if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        r_ = scipy.stats.bernoulli(p=1-self.p0M).rvs(size, random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_]= self.distZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        p_ = np.array(q)
        temp = self.dist.ppf((1 - self.dist.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))
        return temp

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray

        """
        a_ = self.a
        b_ = self.b
        return self.p0M+(1 - self.p0M)*((1 + a_/(a_ - 1)*(f - 1))**(-b_/a_ - 1)\
            - (1 - a_)**(b_/a_ + 1))/(1 - (1 - a_)**(b_/a_ + 1))

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: ``numpy.float64``

        """
        return self.a, self.b, self.p0M

## Zero-truncated geometric
class ZTGeom:
    """
    Zero-truncated geometric distribution. Geometric distribution with no mass (truncated) in 0.

    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) base rv probability in zero.
    :type p0: float
    :param dist: (private) scipy reference non-zero-truncated distribution.
    :type dist: ``scipy.stats._discrete_distns.geom_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          Zero-truncated geometric distribution probability parameter p.

    """
    name = 'ZTgeom'
    def __init__(self, **kwargs):
        self.__p = kwargs['p']
        self.__a = 1 - self.p
        self.__b = 0
        self.__p0 = np.array([self.p])
        self.__dist = scipy.stats.geom(p=self.p)
        
    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be between zero and one.')
        self.__p = value

    @property
    def dist(self):
        return self.__dist

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def p0(self):
        return self.__p0

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """

        temp = (self.dist.pmf(k + 1)) / (1 - self.p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return (self.dist.cdf(k) - self.dist.cdf(0))/(1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        random_state = 1234 if random_state is None else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')
        
        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """

        p_ = np.array(q)
        return (self.dist.ppf(q=p_ * (1 - self.dist.cdf(0)) + self.dist.cdf(0)))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """
        return (1 / (1 - (f - 1) / (1 - self.a)) - 1 + self.a) / self.a

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, 0

## Zero-modified geometric
class ZMGeom:
    """
    Zero-modified geometric distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified geometric distribution.

    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) base rv probability in zero.
    :type p0: float
    :param dist: (private) scipy reference non-zero-modified distribution.
    :type dist: `scipy.stats._discrete_distns.geom_gen``
    :param distZT: (private) reference zero-truncated distribution.
    :type distZT: :py:class:'~ZTGeom'

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          Zero-modified geometric distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          Zero-modified geometric mixing parameter.
    """

    name = 'ZMgeom'
    def __init__(self, **kwargs):
        self.__p = kwargs['p']
        self.__p0M = kwargs['p0M']
        self.__a = 1 - self.p
        self.__b = 0
        self.__p0 = np.array([self.p])
        self.__dist = scipy.stats.geom(p=self.p)
        self.__distZT = ZTGeom(p=self.p)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be between zero and one.')
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p0M must be between zero and one.')
        self.__p0M = value

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b

    @property
    def p0(self):
        return self.__p0

    @property
    def dist(self):
        return self.__dist

    @property
    def distZT(self):
        return self.__distZT

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray
        """
        return (self.dist.pmf(k + 1) * (1 - self.p0M)) / (1 - self.p0)

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: cumulative density function will be computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.p0M + (1 - self.p0M) * (self.dist.cdf(k) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        r_ = scipy.stats.bernoulli(p=1 - self.p0M).rvs(size, random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self.__distZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        p_ = np.array(q)
        return self.dist.ppf((1 - self.dist.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        return self.p0M+(1 - self.p0M)*(1/(1 - (f - 1)/(1 - self.a)) - 1 + self.a)/self.a

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, self.p0M

## Zero-truncated negative binomial
class ZTNegBinom:
    """
    Zero-truncated negative binomial distribution. Negative binomial distribution with no mass (truncated) in 0.

    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: float
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: float
    :param p0: (private) base rv probability in zero.
    :type p0: float
    :param dist: (private) scipy reference non-zero-truncated distribution.
    :type dist: ``scipy.stats._discrete_distns.nbinom_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-truncated negative binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-truncated negative binomial distribution probability parameter p.
    """

    name = 'ZTnbinom'
    def __init__(self, **kwargs):
        self.__n = kwargs['n']
        self.__p = kwargs['p']
        self.__a = 1 - self.p
        self.__b = (self.n - 1) * (1 - self.p)
        self.__p0 = np.array([self.p ** self.n])
        self.__dist = scipy.stats.nbinom(n=self.n, p=self.p)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert isinstance(value, int) and value >= 0, logger.error(
            'n must be a positive number')
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be between zero and one.')
        self.__p = value

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b

    @property
    def p0(self):
        return self.__p0

    @property
    def dist(self):
        return self.__dist

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        temp = (self.dist.pmf(k))/(1 - self.p0)
        x = np.array(k)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return (self.dist.cdf(k) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        random_state = 1234 if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        q_ = np.random.uniform(low=self.dist.cdf(0), high=1, size=size)
        return self.dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        p_ = np.array(q)
        tmp_ = (p_ * (1 - self.dist.cdf(0)) + self.dist.cdf(0))
        return self.dist.ppf(q=tmp_)

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``

        """
        c_ = self.b/self.a+1
        d_ = 1-self.a
        return ((1/(1-(f-1)*self.a/d_))**c_-d_**c_)/(1-d_**c_)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a,b,probability in zero
        :rtype: ``numpy.float64``
        """
        return self.a, self.b, 0

## Zero-modified negative binomial
class ZMNegBinom:
    """
    Zero-modified negative binomial distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified negative binomial distribution.

    :param a: (private) rv a parameter according to the (a, b, k) parametrization.
    :type a: ``float``
    :param b: (private) rv b parameter according to the (a, b, k) parametrization.
    :type b: ``float``
    :param p0: (private) base rv distribution probability mass in zero.
    :type p0: ``float``
    :param dist: (private) scipy reference non-zero-modified distribution.
    :type dist: ``scipy.stats._discrete_distns.nbinom_gen``
    :param distZT: (private) correspondent zero-truncated distribution.
    :type dist: :py:class:'~_ZTNegBinom'

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-truncated negative binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-modified negative binomial distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          Zero-modified negative binomial mixing parameter.

    """
    name = 'ZMnbinom'
    def __init__(self, **kwargs):
        self.__n = kwargs['n']
        self.__p = kwargs['p']
        self.__p0M = kwargs['p0M']
        self.__a = 1 - self.p
        self.__b = (self.n - 1) * (1 - self.p)
        self.__p0 = np.array([(1 / self.p) ** -self.n])
        self.__dist = scipy.stats.nbinom(n=self.n, p=self.p)
        self.__distZT = ZTNegBinom(n=self.n,p=self.p)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        assert isinstance(float(value), float) and value >= 0, logger.error(
            'n must be a positive number')
        self.__n = float(value)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be between zero and one.')
        self.__p = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p must be between zero and one.')
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p0M must be between zero and one.')
        self.__p0M = value

    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b

    @property
    def p0(self):
        return self.__p0

    @property
    def dist(self):
        return self.__dist

    @property
    def distZT(self):
        return self.__distZT

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return (self.dist.pmf(k) * (1 - self.p0M)) / (1 - self.p0)

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.p0M + (1 - self.p0M) * (self.dist.cdf(k) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        random_state = 1234 if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        r_= scipy.stats.bernoulli(p=1-self.p0M).rvs(size, random_state=random_state)
        c_= np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self.distZT.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray
        """

        p_ = np.array(q)
        return self.dist.ppf((1 - self.dist.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the rv given the (a, b, k) parametrization.

        :param f:
        :type f: 
        :return: probability generated in f.
        :rtype: numpy.ndarray
        """

        c_ = 1-self.a
        d_ = (self.b/self.a+1)

        return self.p0M+(1-self.p0M)*((1/(1-(f-1)*self.a/c_))**d_-c_**d_)/(1-c_**d_)

    def abk(self):
        """
        It returns the abk parametrization

        :return: a, b, probability in zero
        :rtype: ``numpy.float64``
        """

        return self.a, self.b, self.p0M

## Zero-modified distrete logarithmic
class ZMLogser:
    """
    Zero-modified (discrete) logarithmic (log-series, series) distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified logarithmic distribution.

    :param p0: base RV probability in zero.
    :type p0: ``float``
    :param dist: base RV distribution.
    :type dist: ``scipy.stats._discrete_distns.logser_gen``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          ZM discrete logarithmic distribution probability parameter p.
        * *p0M* (``numpy.float64``) --
          ZM discrete logarithmic mixing parameter.
    """

    name = 'ZMlogser'
    def __init__(self, **kwargs):
        self.__p = kwargs['p']
        self.__p0M = kwargs['p0M']
        self.__p0 = 0
        self.__dist = scipy.stats.logser(p=self.p)

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p has to be between zero and one.')
        self.__p = value

    @property
    def p0M(self):
        return self.__p0M

    @p0M.setter
    def p0M(self, value):
        assert value >= 0 and value <= 1, logger.error(
            'p0M must be between zero and one.')
        self.__p0M = value

    @property
    def p0(self):
        return self.__p0

    @property
    def dist(self):
        return self.__dist

    def pmf(self, k):
        """
        Probability mass function.

        :param k: quantile where probability mass function is evaluated.
        :type k: int

        :return: probability mass function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return (self.dist.pmf(k) * (1 - self.p0M)) / (1 - self.p0)

    def logpmf(self, k):
        """
        Natural logarithm of the probability mass function.

        :param k: quantile where the (natural) probability mass function logarithm is evaluated.
        :type k: int

        :return: natural logarithm of the probability mass function
        :rtype: numpy.float64 or numpy.ndarray
        """

        return np.log(self.pmf(k))

    def cdf(self, k):
        """
        Cumulative distribution function.

        :param k: quantile where the cumulative distribution function is evaluated.
        :type k: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.p0M + (1 - self.p0M) * (self.dist.cdf(k) - self.dist.cdf(0)) / (1 - self.dist.cdf(0))

    def logcdf(self, k):
        """
        Log of the cumulative distribution function.

        :param k: quantile where log of the cumulative density function is evaluated.
        :type k: int
        :return: natural logarithm of the cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return np.log(self.cdf(k=k))
   
    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size.
        :type size: int, optional
        :param random_state: random state for the random number generator.
        :type random_state: int, optional
        :return: random variates.
        :rtype: numpy.int or numpy.ndarray

        """
        random_state = 1234 if (random_state is None) else random_state
        assert isinstance(random_state, int), logger.error("random_state has to be an integer")
        np.random.seed(random_state)

        try:
            size = int(size)
        except:
            logger.error('Please provide size as an integer')

        q_ = np.random.uniform(0, 1, size)
        return self.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cdf.

        :param q: level at which the percent point function is evaluated.
        :type q: float
        :return: percent point function.
        :rtype: numpy.float64 or numpy.ndarray
        """

        p_ = np.array(q)
        temp = self.dist.ppf((1 - self.dist.cdf(0)) * (p_ - self.p0M) / (1 - self.p0M) + self.dist.cdf(0))

        zeros = np.where(p_ <= self.p0M)[0]
        if zeros.size == 0:
            return temp
        else:
            if p_.shape == ():
                temp = self.p0M
            else:
                temp[zeros] = self.p0M
            return temp

## Exponential
class Exponential:
    """
    expontential distribution.

    :param theta: exponential distribution theta parameter.
    :type theta: ``float``
    :param loc: location parameter
    :type loc: ``float``

    """
    name='exponential'
    def __init__(self, loc=0, theta=1):
        self.theta = theta
        self.loc=loc
        #scipy exponential distribution
        self.dist = scipy.stats.expon(loc=self.loc)

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, var):
        assert var > 0 , logger.error(
            'theta must be positive')
        self.__theta = var

    def pdf(self,x):
        """
        Probability density function.

        :param x: the probability density function will be computed in x.
        :type x:``numpy.ndarray``
        :return: pdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.theta*self.dist.pdf(self.loc+self.theta*x)

    def logpdf(self,x):
        """
        Log of the probability density function.

        :param x: the log of the probability function will be computed in x.
        :type x:``numpy.ndarray``
        :return: logpdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logpdf(self.theta*x)

    def cdf(self,x):
        """
        Cumulative distribution function.

        :param x: the cumulative distribution function will be computed in x.
        :type x: ``numpy.ndarray``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.cdf(self.loc+self.theta * (x-self.loc))

    def logcdf(self,x):
        """
        Log of the cumulative distribution function.

        :param x: log of the cumulative density function computed in k.
        :type x: ``int``
        :return: cdf`
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logcdf(self.theta*x)

    def sf(self,x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in k.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.sf(self.theta*x)

    def logsf(self,x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.logsf(self.theta*x)

    def isf(self,x):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :type q: ``numpy.ndarray``
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.dist.isf(self.theta*x)

    def rvs(self, size,random_state=42):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')
        return scipy.stats.expon.rvs(size=size,random_state=random_state)/self.theta+self.loc

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: (differential) entropy.
        :rtype: ``numpy.float64``
        """
        return 1-np.log(self.theta)

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return 1/ self.theta

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return 1/ self.theta**2

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return np.sqrt(self.variance)

    def ppf(self,x):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :type q: ``numpy.ndarray``
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        try:
            x=np.array(x)
            assert isinstance(x, np.ndarray), logger.error('x values must be an array')
        except:
            logger.error('Please provide the x quantiles you want to evaluate as an array')

        temp= -np.log(1-x)/self.theta

        zeros = np.where(((x>=1.)& (x <=0.)))[0]

        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = np.float64('nan')
            else:
                temp[zeros] = np.float64('nan')
            return temp

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        v = np.array([v])
        out = (1 - np.exp(-self.theta * v)) / self.theta
        out[v < 0] = v[v < 0]
        return out

    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        return (1 - scipy.stats.expon.cdf(self.theta * (d - loc)))

## Gamma
class Gamma:
    """
    Wrapper to scipy gamma distribution.
    When a is an integer it reduces to an Erlang distribution.
    When a=1 it reduces to an Exponential distribution.

    :param a: gamma shape parameter a.
    :type a: ``int`` or ``float``
    :param beta: inverse of the gamma scale parameter.
    :type beta: ``float``
    :param scale: gamma scale parameter.
    :type scale: ``float``
    :param loc: gamma location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.gamma_gen ``
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.a=kwargs['a']
        self.scale = scale
        self.loc = loc
        self.__dist = scipy.stats.gamma(a=self.a, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: int
        :param random_state: random state for the random number generator.
        :type random_state: int

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size,random_state=random_state)

    def pdf(self,x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: int

        :return: pdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self,x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: int
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self,x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self,x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in k.
        :type k: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self,x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in k.
        :type x: int
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self,x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in k.
        :type x: int
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self,q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self,q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self,n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self,data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)


    def expect(self,func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self,alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self,v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v])
        try:
            beta = 1 / self.scale
        except:
            beta = 1
        alpha = self.a
        out = (alpha / beta) * scipy.special.gammainc(alpha + 1, beta * v) + v * (
                1 - scipy.special.gammainc(alpha, beta * v))
        out[v < 0] = v[v < 0]
        return out

    def den(self,d,loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            beta = 1 / self.spar['scale']
        except:
            beta = 1

        return 1 - scipy.stats.gamma(loc=0,a=self.a,scale=1/beta).cdf(d - loc)

## Generalized Pareto
class GenPareto:
    """
    Wrapper to scipy genpareto distribution.
    When c=0 it reduces to an Exponential distribution.
    When c=-1 it reduces to a uniform distribution.
    When the correct parametrization is adopted, it is possible to fit all the Pareto types.

    :param c: genpareto shape parameter c.
    :type c: `` float``
    :param scale: genpareto scale parameter.
    :type scale: ``float``
    :param loc: genpareto location parameter.
    :type loc:``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.genpareto_gen ``

    """
    name="genpareto"
    def __init__(self, loc=0, scale=1, **kwargs):
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc
        self.__dist = scipy.stats.genpareto(c=self.c, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v])
        try:
            scale_ = self.scale
        except:
            scale_ = 1
        out = (scale_ / (self.c - 1)) * ((1 + self.c * v / scale_) ** (1 - 1 / self.c) - 1)
        out[v < 0] = v[v < 0]
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            scale_ = self.scale
        except:
            scale_ = 1

        return 1 - scipy.stats.genpareto(loc=0,c=self.c,scale=scale_).cdf(d - loc)

#### Lognormal
class Lognorm:
    """
    Wrapper to scipy lognormal distribution.

    :param s: lognormal shape parameter s.
    :type s: `` float``
    :param scale: lognormal scale parameter.
    :type scale: ``float``
    :param loc: lognormal location parameter.
    :type loc:``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.lognorm_gen ``
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.s = kwargs['s']
        self.scale = scale
        self.loc = loc
        self.__dist = scipy.stats.lognorm(s=self.s, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v])
        out = v.copy()
        try:
            loc = np.log(self.scale)
        except:
            loc = 0
        shape = self.s
        out[v > 0] = np.exp(loc + shape ** 2 / 2) * (scipy.stats.norm.cdf((np.log(v[v > 0]) - (loc + shape ** 2)) / shape)) + \
                     v[v > 0] * (
                             1 - scipy.stats.norm.cdf((np.log(v[v > 0]) - loc) / shape))
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """
        try:
            loc_ = np.log(self.scale)
        except:
            loc_ = 0
        return 1 - scipy.stats.lognorm(scale=np.exp(loc_), s=self.s).cdf(d - loc)

# Generalized beta
class GenBeta:
    """
    Generalized Beta (GB) distribution, also refer to as Generalized Beta
    of the second kind, or the Generalized Beta Prime distribution. 
    If X is a GB distributed r.v., its cumulative distribution function can
    be expressed as:
    
    Pr[X <= x] = Pr[Y <= (x/scale)^shape3], 0 < x < scale,

    where Y has a Beta distribution, with parameters shape1 and shape2.
    Refer to Appendix A of Klugman, Panjer & Willmot, Loss Models, Wiley.
    """
    def __init__(self, shape1, shape2, shape3, scale):
        
        assert shape1 > 0, "shape1 has to be > 0"
        assert shape2 > 0, "shape2 has to be > 0"
        assert shape3 > 0, "shape3 has to be > 0"
        assert scale > 0, "scale has to be > 0"

        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.scale = scale

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        try:
            size=int(size)
        except:
            logger.error('Please provide size as an integer')

        random_state = 1234 if (random_state is None) else random_state
        np.random.seed(random_state)

        tmp_ = scipy.stats.beta(a=self.shape1, b=self.shape2).rvs(size=size, random_state=random_state)
        return self.scale * pow(tmp_, 1.0/self.shape3)

    def pdf(self, x):
        """
        probability density function.

        :param x: probability density function will be computed in x.
        :type x: ``int``
        :return: pdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        if (x < 0.0 or x > self.scale):
            return 0
        
        psh = self.shape1 * self.shape3

        if (x == 0.0):
            if (psh > 1):
                return(0)
            elif (psh < 1):
                return np.infty
            else:
                return self.shape3/ scipy.special.beta(self.shape1, self.shape2)
        
        if (x == self.scale):
            if (self.shape2 > 1):
                return 0
            if (self.shape2 < 1):
                return np.infty 
            else:
                return self.shape1 * self.shape3

        logu = self.shape3 * (np.log(x) - np.log(self.scale))
        log1mu = np.log1p(-np.exp(logu))

        return np.exp(np.log(self.shape3) + self.shape1 * logu + (self.shape2 - 1.0) * \
            log1mu - np.log(x) - scipy.special.betaln(self.shape1, self.shape2))

    def cdf(self, x):

        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in x.
        :type x: ``real``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        if (x <= 0):
            return 0
        if (x >= self.scale):
            return 1

        u = np.exp(self.shape3 * (np.log(x) - np.log(self.scale)))
        return self.__dist.cdf(x=u)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.pdf(x=x))

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x=x))

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return 1 - self.cdf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.sf(x=x))

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.scale * pow(self.__dist.ppf(q=q), 1.0/self.shape3)

    # def isf(self, q):
    #     """
    #     Inverse survival function (inverse of sf).

    #     :param q: Inverse survival function computed in q.
    #     :return: inverse sf
    #     :rtype: ``numpy.float64`` or ``numpy.ndarray``
    #     """
    #     return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        if (n <= self.shape1 * self.shape3):
            return np.inf
        tmp_ = n / self.shape3

        return pow(self.scale, n) * scipy.special.beta(self.shape1 + tmp_, self.shape2) / \
            scipy.special.beta(self.shape1, self.shape2)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        t_ = []
        if 'm' in moments:
            t_.append(self.mean())
        if 'v' in moments:
            t_.append(self.var())
        if 's' in moments:
            t_.append(self.moment(3) / self.moment(2)**(3/2))
        if 'k' in moments:
            t_.append(self.moment(4) / self.moment(2)**2 - 3)
        assert len(t_) > 0, "moments argument is not composed of letters 'mvsk'"

        return tuple(t_)

    # def entropy(self):
    #     """
    #     (Differential) entropy of the RV.

    #     :return: entropy
    #     :rtype: ``numpy.ndarray``
    #     """
    #     return self.__dist.entropy()

    # def fit(self, data):
    #     """
    #     Parameter estimates for generic data.

    #     :param data: data on which to fit the distribution.
    #     :return: fitted distribution.
    #     """
    #     return self.__dist.fit(data)

    # def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
    #     """
    #     Expected value of a function (of one argument) with respect to the distribution.

    #     :param func: class 'function'.
    #     :param args:argument of func.
    #     :param lb: lower bound.
    #     :param ub: upper bound.
    #     :return: expectation with respect to the distribution.

    #     """
    #     if args is None:
    #         args = (self.a,)
    #     return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.ppf(p=0.5)

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.self.moment(1)

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.moment(2) - self.moment(1)**2

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.var()**(1/2)

    # def interval(self, alpha):
    #     """
    #     Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

    #     :param alpha: fraction alpha
    #     :rtype alpha: float
    #     :return: Endpoints
    #     :rtype: tuple
    #     """
    #     return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        if (1 <= - self.shape1 * self.shape3):
            return np.inf
        
        if (v <= 0.0):
            return 0.0
        
        z_ = 0 if np.isinf(v) else v
        
        tmp_ = 1 / self.shape3
        u_ = np.exp(self.shape3 * (np.log(v) - np.log(self.scale)))

        return self.scale * scipy.special.beta(self.shape1 + tmp_, self.shape2) / \
            scipy.special.beta(self.shape1, self.shape2) * scipy.stats.beta.cdf(u_, self.shape1 + tmp_, self.shape2) \
            + z_ * self.__dist.cdf(u_)

    # def den(self, d, loc):
    #     """
    #     It returns the denominator of the local moments discretization.

    #     :param d: lower priority.
    #     :type d: ``float``
    #     :param loc: location parameter.
    #     :type loc: ``float``
    #     :return: denominator to compute the local moments discrete sequence.
    #     :rtype: ``numpy.ndarray``
    #     """
    #     try:
    #         scale_ = self.scale
    #     except:
    #         scale_ = 1

    #     return 1 - stats.genpareto(loc=0,c=self.c,scale=scale_).cdf(d - loc)

class Burr12:
    """
    Wrapper to scipy burr distribution.
    It is referred to the Burr Type XII, Singhâ€“Maddala distribution.
    When d=1, this is a Fisk distribution.
    When c=d, this is a Paralogistic distribution.

    :param c: burr shape parameter c.
    :type c: `` float``
    :param d: burr shape parameter d.
    :type d: `` float``
    :param scale: burr scale parameter.
    :type scale: ``float``
    :param loc: burr location parameter.
    :type loc: ``float``
    :param __dist: scipy corresponding RV.
    :type __dist: ``scipy.stats._continuous_distns.lognorm_gen ``


    """

    def __init__(self, loc=0, scale=1, **kwargs):
        self.c = kwargs['c']
        self.d = kwargs['d']
        self.scale = scale
        self.loc = loc
        self.__dist = scipy.stats.burr12(c=self.c,
                                    d=self.d,
                                    loc=self.loc,
                                    scale=self.scale)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size.
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.rvs(size=size, random_state=random_state)

    def pdf(self, x):

        """
        Probability distribution function

        :param x: probability distribution function will be computed in k.
        :type x: ``int``

        :return: pdf
        :rtype:``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.pdf(x=x)

    def logpdf(self, x):
        """
        Log of the probability distribution function.

        :param x: log of the probability distribution function computed in k.
        :type x: ``int``
        :return: log pmf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.logpdf(x=x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in k.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.cdf(x=x)

    def logcdf(self, x):
        """
        Log of the cumulative distribution function.

        :param k: log of the cumulative density function computed in x.
        :type k: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logcdf(x=x)

    def sf(self, x):
        """
        Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

        :param x: survival function will be computed in x.
        :type x: ``int``
        :return: sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.__dist.sf(x=x)

    def logsf(self, x):
        """
        Log of the survival function.

        :param x:log of the survival function computed in x.
        :type x: ``int``
        :return: cdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.logsf(x=x)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf â€” percentiles).

        :param q: Percent point function computed in q.
        :return: ppf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.__dist.isf(q=q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment of order n
        :return: non-central moment of order n
        """
        return self.__dist.moment(n=n)

    def stats(self, moments='mv'):
        """
        Mean(â€˜mâ€™), variance(â€˜vâ€™), skew(â€˜sâ€™), and/or kurtosis(â€˜kâ€™).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        return self.__dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self.__dist.entropy()

    def fit(self, data):
        """
        Parameter estimates for generic data.

        :param data: data on which to fit the distribution.
        :return: fitted distribution.
        """
        return self.__dist.fit(data)

    def expect(self, func, args=None, lb=None, ub=None, conditional=False, **kwds):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: class 'function'.
        :param args:argument of func.
        :param lb: lower bound.
        :param ub: upper bound.
        :return: expectation with respect to the distribution.

        """
        if args is None:
            args = (self.a,)
        return self.__dist.expect(func, args=args, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.__dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.__dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.__dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.__dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :rtype alpha: float
        :return: Endpoints
        :rtype: tuple
        """
        return self.__dist.interval(alpha=alpha)

    def Emv(self, v):
        """
        Expected value of the function min(x,v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.ndarray``
        :return: expected value of the minimum function.
        """
        v = np.array([v]).flatten()
        out = v.copy()
        u = v.copy()
        u[v>0] = 1 / (1 + (v[v>0] / self.scale) ** self.c)
        out[v>0] = v[v>0]*(u[v>0]**self.d)+scipy.special.betaincinv(1+1/self.c,self.d-1/self.c,1-u[v>0])*(self.scale*scipy.special.gamma(1+1/self.c)*scipy.special.gamma(self.d-1/self.c)/scipy.special.gamma(self.d))
        return out


    def den(self, d, loc):
        """
        It returns the denominator of the local moments discretization.

        :param d: lower priority.
        :type d: ``float``
        :param loc: location parameter.
        :type loc: ``float``
        :return: denominator to compute the local moments discrete sequence.
        :rtype: ``numpy.ndarray``
        """

        return 1 - scipy.stats.burr12(c=self.c,d=self.d,scale=self.scale).cdf(d - loc)