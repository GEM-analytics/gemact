import numpy as np
from scipy.special import factorial
import helperfunctions as hf
import distributions as distributions
from twiggy import quick_setup,log

quick_setup()
logger= log.name('lossaggregation')

DCEILING = 5
DIST_DICT = {
    'gamma': distributions.Gamma,
    'lognorm': distributions.Lognorm,
    'exponential':distributions.Exponential,
    'genpareto': distributions.GenPareto,
    'burr12': distributions.Burr12
    #   'dagum': distributions.Dagum,
    #   'invgamma': distributions.Invgamma,
    #   'weibull_min': distributions.Weibull_min,
    #   'invweibull': distributions.Invweibull,
    #  'beta': distributions.Beta
    }

COP_DICT = {
    'clayton': hf.ClaytonCDF,
    'frank': hf.FrankCDF,
    'gumbel':hf.GumbelCDF
    }

class LossAggregation:
    """
        This class computes the probability of the sum of random variables
        with a dependence structure specified by a copula via the AEP algorithm.
    """
    def __init__(self, **kwargs):
        
        global DCEILING
        global DIST_DICT
        global COP_DICT
        
        # properties
        self.copula = kwargs['copula']
        self.copula_par = kwargs['copula_par']
        self.margins = kwargs['margins']
        self.margins_pars = kwargs['margins_pars']
        
        # private attributes
        self.__d = len(self.margins)
        self.__a = 2. / (self.__d + 1) # Alpha parameter of the AEP algorithm.
        self.__b = np.repeat(0, self.__d).reshape(1, self.__d) # Vector b of the AEP algorithm.
        self.__h = None # Vector h of the AEP algorithm.
        self.__sn = np.array([1]) # Array of +1,-1, 0 indicating whether a volume must be summed, subtracted or ignored, respectively.
        self.__n_simpleces = 2**self.__d - 1 # Number of new simpleces received in each step.
        self.__mat = hf.cartesian_product(*([np.array([0, 1])]*self.__d)).T # Matrix of the vectors in the {0,1}**d_la space.
        self.__card = np.sum(self.__mat, axis=1)[1:] # Cardinality of the 'mat' matrix.
        self.__m = self.m_j() # Array of +1, -1, 0, indicating whether the new simpleces origined from sn must be summed, subtracted or ignored, respectively.
        self.__s = (-1) ** (self.__d - np.sum(self.__mat, axis=1)) # Array of +1 or -1, indicating whether to sum or subtract a volume, respectively.
        self.__ext = ((self.__d + 1)**self.__d)/(factorial(self.__d)*2**self.__d) # implement its usage

    @property
    def margins_pars(self):
        return self.__margins_pars

    @margins_pars.setter
    def margins_pars(self, value):
        assert isinstance(value, list), logger.error("Please provide a list")
        assert len(value) == len(self.margins), logger.error("Margins and margins_pars must have the same dimension")

        for j in range(len(value)):
            assert isinstance(value[j], dict), logger.error("Please provide a list of dictionaries")

            try:
                DIST_DICT[self.margins[j]](**value[j])
            except:
                logger.error('The marginal distribution %r is not parametrized correctly.' %j)

        self.__margins_pars = value

    @property
    def margins(self):
        return self.__margins

    @margins.setter
    def margins(self, value):
        assert isinstance(value, list), logger.error("Please provide a list")

        assert len(value) <= DCEILING, logger.error("Number of dimensions exceeds limit of %r" % DCEILING)

        for j in range(len(value)):
            assert value[j] in DIST_DICT.keys(), "%r distribution is not supported. \n See https://gem-analytics.github.io/gemact/" % value[j]

        self.__margins = value

    @property
    def copula(self):
        return self.__copula

    @copula.setter
    def copula(self, value):
        assert isinstance(value, str), logger.error('Copula name must be given as a string')
        assert value in COP_DICT.keys(), "%r copula is not supported.\n See https://gem-analytics.github.io/gemact/" % value
        self.__copula = value

    @property
    def copula_par(self):
        return self.__copula_par

    @copula_par.setter
    def copula_par(self, value):
        assert isinstance(value, dict), 'The copula distribution parameters must be given as a dictionary'

        try:
            COP_DICT[self.copula](**value)
        except:
            logger.error('Copula not correctly parametrized.\n See https://gem-analytics.github.io/gemact/ ')

        self.__copula_par = value

    def copula_cdf(self, k):
        result = COP_DICT[self.copula](**self.copula_par).cdf(k.transpose())
        return np.array(result)

    def margins_cdf(self, k):
        result = [DIST_DICT[self.margins[j]](**self.margins_pars[j]).cdf(k[j, :]) for j in range(self.__d)]
        return np.array(result)

    def volume_calc(self):
        mat_ = np.expand_dims(self.__mat, axis=2)
        h_ = self.__a * self.__h
        b_ = np.expand_dims(self.__b.T, axis=0)
        s_ = self.__s.reshape(-1, 1)
        v_ = np.hstack((b_ + h_*mat_)) # np.dstack(zip( (b_ + h_*mat_) ))[0]    # <- WARNING 
        c_ = self.copula_cdf(self.margins_cdf(v_)).reshape(-1, self.__b.shape[0])
        result = np.sum(c_ * (s_ * np.sign(h_)**self.__d), axis=0)
        return result

    def m_j(self):
        result = self.__card.copy()
        greater = np.where(result > (1 / self.__a))
        equal = np.where(result == (1 / self.__a))
        lower = np.where(result < (1 / self.__a))
        result[greater] = (-1) ** (self.__d + 1 - result[greater])
        result[equal] = 0
        result[lower] = (-1) ** (1 + result[lower])
        return result

    def sn_update(self):
        result = np.repeat(self.__sn, self.__n_simpleces) * np.tile(self.__m, self.__sn.shape[0])
        return result

    def h_update(self):
        result = (1 - np.tile(self.__card, len(self.__h)) * self.__a) * np.repeat(self.__h, len(self.__card))
        return result

    def b_update(self):
        mat_ = self.__mat[1:, :].transpose()
        h_ = np.repeat(self.__h, self.__n_simpleces).reshape(-1, 1)
        times_ = int(h_.shape[0] / mat_.shape[1])
        result = np.repeat(self.__b, self.__n_simpleces, 0)
        result = result + self.__a * np.tile(h_, (1, self.__d)) * np.tile(mat_, times_).transpose()
        return result 

    def cdf(self, k, n_iter=7):
        # method="AEP"
        self.__h = np.array([[k]]) # Vector h of the AEP algorithm.
        result = self.volume_calc()[0]
        for _ in range(n_iter):
            self.__sn = self.sn_update()
            self.__b = self.b_update()
            self.__h = self.h_update()
            result += np.sum(self.__sn * self.volume_calc())
        return result

mylaggr = LossAggregation(
    margins=['genpareto', 'genpareto', 'genpareto'],
    margins_pars=[
        {'loc':0, 'scale':1/0.9, 'c':1/0.9},
        {'loc':0, 'scale':1/1.8, 'c':1/1.8},
        {'loc':0, 'scale':1/1.5, 'c':1/1.5},
        ],
    copula='gumbel',
    copula_par={'par':1.2}
    )
# #output is kept in the out attribute
mylaggr.cdf(2)
# print(la.out)