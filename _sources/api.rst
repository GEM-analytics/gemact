API reference guide
===================

This page documents the GEMAct API exposed by the main modelling modules.
The class entries below intentionally avoid properties and private implementation classes so that the rendered API focuses on constructors, parameters, and documented methods.

``LossModel``
-------------

Risk costing
~~~~~~~~~~~~

GEMAct costing model is based on the collective risk theory. The aggregate loss :math:`X`, also referred to as aggregate claim cost, is

.. math::
   :label: crm

   X = \sum_{i=1}^{N} Z_i,

where the following assumptions hold:

* :math:`N` is a random variable taking values in :math:`\mathbb{N}_0` representing the claim frequency.

* :math:`\left\{ Z_i\right\}_{i \in \mathbb{N}}` is a sequence of i.i.d. non-negative random variables independent of :math:`N`; :math:`Z` is the random variable representing the individual claim loss.

Equation :eq:`crm` is often referred to as the frequency-severity loss model representation.
This can encompass common coverage modifiers present in (re)insurance contracts. More specifically, we consider:

* For :math:`a \in [0, 1]`, the function :math:`Q_a` apportioning the aggregate loss amount:

  .. math::

     Q_a (X)= a X.

* For :math:`c,d \geq 0`, the function :math:`L_{c, d}` applied to the individual claim loss:

  .. math::
     :label: minmax

     L_{c, d} (Z_i) = \min \left\{\max \left\{0, Z_i-d\right\}, c\right\}.

  Herein, for each and every loss, the excess to a deductible :math:`d`, sometimes referred to as priority,
  is considered up to a cover or limit :math:`c`.
  Similarly to the individual loss :math:`Z_i`, Formula :eq:`minmax` can be applied to the aggregate loss :math:`X`.

The expected value of the aggregate loss constitutes the building block of an insurance tariff.
Listed below are some examples of basic reinsurance contracts whose pure premium can be computed with GEMAct.

* The Quota Share (QS), where a share :math:`a` of the aggregate loss is ceded to the reinsurer together with the respective premium, and the remaining part is retained:

  .. math::

     \text{P}^{QS} = \mathbb{E}\left[ Q_a \left( X \right)\right].

* The Excess-of-loss (XL), where the insurer cedes to the reinsurer each and every loss exceeding a deductible :math:`d`, up to an agreed limit or cover :math:`c`, with :math:`c,d \geq 0`:

  .. math::

     \text{P}^{XL} =  \mathbb{E}\left[ \sum_{i=1}^{N} L_{c,d} (Z_i) \right].

* The Stop Loss (SL), where the reinsurer covers the aggregate loss exceedance of an aggregate deductible :math:`v`, up to an aggregate limit or cover :math:`u`, with :math:`u,v \geq 0`:

  .. math::

     \text{P}^{SL} = \mathbb{E}\left[ L_{u, v} (X) \right].

* The Excess-of-loss with reinstatements (RS) (:cite:t:`sundt`). Assuming the aggregate cover :math:`u` is equal to :math:`(K + 1)c`, with :math:`K \in \mathbb{Z}^+`:

  .. math::
     :label: rs

     \text{P}^{RS} =  \frac{\mathbb{E}\left[ L_{u, v} (X) \right]}{1+\frac{1}{c} \sum_{k=1}^K l_k \mathbb{E}\left[ L_{c, (k-1)c+v}(X) \right]}.

  Where :math:`K` is the number of reinstatement layers and :math:`l_k \in [0, 1]` is the reinstatement premium percentage, with :math:`k=1, \ldots, K`.
  When :math:`l_k = 0`, the :math:`k`-th reinstatement is said to be free.

The following table gives the correspondence between the ``LossModel`` class attributes and the costing model above.

+------------------------+-----------------------------------------+
| Costing model notation | Parametrization in ``LossModel``        |
+========================+=========================================+
| :math:`d`              | ``deductible``                          |
+------------------------+-----------------------------------------+
| :math:`c`              | ``cover``                               |
+------------------------+-----------------------------------------+
| :math:`v`              | ``aggr_deductible``                     |
+------------------------+-----------------------------------------+
| :math:`u`              | ``aggr_cover``                          |
+------------------------+-----------------------------------------+
| :math:`K`              | ``n_reinst``                            |
+------------------------+-----------------------------------------+
| :math:`l_k`            | ``reinst_percentage``                   |
+------------------------+-----------------------------------------+
| :math:`\alpha`         | ``share``                               |
+------------------------+-----------------------------------------+

For additional information the reader can refer to :cite:t:`b:kp`, :cite:t:`sundt`.
Further details on the computational methods to approximate the aggregate loss distribution can be found in :cite:t:`b:kp` and :cite:t:`embrechts`.

Example
^^^^^^^

Below is an example of costing an XL contract with reinstatements::

   from gemact.lossmodel import Frequency, Severity, PolicyStructure, Layer, LossModel

   lossmodel_RS = LossModel(
       frequency=Frequency(
           dist='poisson',
           par={'mu': 1.5}
       ),
       severity=Severity(
           par={'loc': 0, 'scale': 83.34, 'c': 0.834},
           dist='genpareto'
       ),
       policystructure=PolicyStructure(
           layers=Layer(
               cover=100,
               deductible=0,
               aggr_deductible=100,
               reinst_percentage=0.5,
               n_reinst=2
           )
       ),
       aggr_loss_dist_method='fft',
       sev_discr_method='massdispersal',
       n_aggr_dist_nodes=int(100000)
   )
   lossmodel_RS.print_costing_specs()

Severity discretization
~~~~~~~~~~~~~~~~~~~~~~~

When passing from a continuous distribution to an arithmetic distribution, it is important to preserve the distribution properties, either locally or globally.
Given a bandwidth, or discretization step, :math:`h` and a number of nodes :math:`M`, in :cite:t:`b:kp` the method of mass dispersal and the method of local moments matching work as follows.

**Method of mass dispersal**

.. math::
   :label: md1

   f_{0}=\operatorname{Pr}\left(Y<\frac{h}{2}\right)=F_{Y}\left(\frac{h}{2}-0\right)

.. math::
   :label: md2

   f_{j}=F_{Y}\left(j h+\frac{h}{2}-0\right)-F_{Y}\left(j h-\frac{h}{2}-0\right), \quad j=1,2, \ldots, M-1

.. math::
   :label: md3

   f_{M}=1-F_{X}[(M-0.5) h-0]

**Method of local moments matching**

The following approach is applied to preserve the global mean of the distribution.

.. math::
   :label: lm1

   f_0 = m^0_0

.. math::
   :label: lm2

   f_j = m^{j}_0+ m^{j-1}_1, \quad j=0,1, \ldots, M

.. math::
   :label: lm3

   \sum_{j=0}^{1}\left(x_{k}+j h\right)^{r} m_{j}^{k}=\int_{x_{k}-0}^{x_{k}+ h-0} x^{r} d F_{X}(x), \quad r=0,1

.. math::
   :label: lm4

   m_{j}^{k}=\int_{x_{k}-0}^{x_{k}+p h-0} \prod_{i \neq j} \frac{x-x_{k}-i h}{(j-i) h} d F_{X}(x), \quad j=0,1

In addition to these two methods, GEMAct also provides upper and lower discretizations.

Example
^^^^^^^

An example of code to implement severity discretization is given below::

    from gemact.lossmodel import Severity
    import numpy as np

    severity = Severity(
        par={'loc': 0, 'scale': 83.34, 'c': 0.834},
        dist='genpareto'
    )

    massdispersal = severity.discretize(
        discr_method='massdispersal',
        n_discr_nodes=50000,
        discr_step=.01,
        deductible=0
    )

    localmoments = severity.discretize(
        discr_method='localmoments',
        n_discr_nodes=50000,
        discr_step=.01,
        deductible=0
    )

    meanMD = np.sum(massdispersal['sev_nodes'] * massdispersal['fj'])
    meanLM = np.sum(localmoments['sev_nodes'] * localmoments['fj'])

    print('Original mean: ', severity.model.mean())
    print('Mean (mass dispersal): ', meanMD)
    print('Mean (local moments): ', meanLM)

Classes
~~~~~~~~~~~~~~

``PolicyStructure``
^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.lossmodel.PolicyStructure
   :no-members:

.. automethod:: gemact.lossmodel.PolicyStructure.index_to_layer_name

.. automethod:: gemact.lossmodel.PolicyStructure.layer_name_to_index

``Layer``
^^^^^^^^^

.. autoclass:: gemact.lossmodel.Layer
   :no-members:

.. automethod:: gemact.lossmodel.Layer.specs

``LayerTower``
^^^^^^^^^^^^^^

.. autoclass:: gemact.lossmodel.LayerTower
   :no-members:

.. automethod:: gemact.lossmodel.LayerTower.append

.. automethod:: gemact.lossmodel.LayerTower.insert

.. automethod:: gemact.lossmodel.LayerTower.extend

.. automethod:: gemact.lossmodel.LayerTower.sort

``Frequency``
^^^^^^^^^^^^^

.. autoclass:: gemact.lossmodel.Frequency
   :no-members:

.. automethod:: gemact.lossmodel.Frequency.abp0g0

``Severity``
^^^^^^^^^^^^

.. autoclass:: gemact.lossmodel.Severity
   :no-members:

.. automethod:: gemact.lossmodel.Severity.excess_frequency

.. automethod:: gemact.lossmodel.Severity.return_period

.. automethod:: gemact.lossmodel.Severity.censored_var

.. automethod:: gemact.lossmodel.Severity.censored_std

.. automethod:: gemact.lossmodel.Severity.censored_mean

.. automethod:: gemact.lossmodel.Severity.censored_skewness

.. automethod:: gemact.lossmodel.Severity.censored_coeff_variation

.. automethod:: gemact.lossmodel.Severity.discretize

.. automethod:: gemact.lossmodel.Severity.plot_discr_sev_cdf

``LossModel``
^^^^^^^^^^^^^

.. autoclass:: gemact.lossmodel.LossModel
   :no-members:

.. automethod:: gemact.lossmodel.LossModel.dist_calculate

.. automethod:: gemact.lossmodel.LossModel.moment

.. automethod:: gemact.lossmodel.LossModel.ppf

.. automethod:: gemact.lossmodel.LossModel.cdf

.. automethod:: gemact.lossmodel.LossModel.sf

.. automethod:: gemact.lossmodel.LossModel.rvs

.. automethod:: gemact.lossmodel.LossModel.mean

.. automethod:: gemact.lossmodel.LossModel.var

.. automethod:: gemact.lossmodel.LossModel.std

.. automethod:: gemact.lossmodel.LossModel.skewness

.. automethod:: gemact.lossmodel.LossModel.coeff_variation

.. automethod:: gemact.lossmodel.LossModel.costing

.. automethod:: gemact.lossmodel.LossModel.print_costing_specs

.. automethod:: gemact.lossmodel.LossModel.print_aggr_loss_method_specs

.. automethod:: gemact.lossmodel.LossModel.print_policy_layer_specs

.. automethod:: gemact.lossmodel.LossModel.plot_dist_cdf


``LossReserve``
---------------

Claims reserving
~~~~~~~~~~~~~~~~

GEMAct provides a software implementation of average cost methods for claims reserving based on the collective risk model framework.
The methods implemented are the Fisher-Lange method in :cite:t:`fisher99` and the collective risk model for claims reserving in :cite:t:`ricotta16`.

It allows for tail estimates and assumes the triangular inputs to be provided as a ``numpy.ndarray`` with two equal dimensions ``(I,J)``, where ``I=J``.
The aim of average cost methods is to model incremental payments as in equation :eq:`acmethods`.

.. math::
   :label: acmethods

   P_{i,j}=n_{i,j} \cdot m_{i,j}

where :math:`n_{i,j}` is the number of payments in the cell :math:`i,j` and :math:`m_{i,j}` is the average cost in the cell :math:`i,j`.

Example
^^^^^^^

It is possible to use the module ``gemdata`` to test GEMAct average cost methods::

    from gemact import gemdata

    ip_ = gemdata.incremental_payments
    in_ = gemdata.incurred_number
    cp_ = gemdata.cased_payments
    cn_ = gemdata.cased_number
    reported_ = gemdata.reported_claims
    claims_inflation = gemdata.claims_inflation

An example of Fisher-Lange implementation::

    from gemact.lossreserve import AggregateData, ReservingModel, LossReserve
    from gemact import gemdata

    ad = AggregateData(
        incremental_payments=gemdata.incremental_payments,
        cased_payments=gemdata.cased_payments,
        payments_number=gemdata.payments_number,
        open_claims_number=gemdata.open_number,
        reported_claims=gemdata.reported_claims
    )

    rm = ReservingModel(
        tail=True,
        reserving_method="fisher_lange",
        claims_inflation=gemdata.claims_inflation
    )

    lr = LossReserve(data=ad, reservingmodel=rm)

Classes
~~~~~~~~~~~~~~

``AggregateData``
^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.lossreserve.AggregateData
   :no-members:

``ReservingModel``
^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.lossreserve.ReservingModel
   :no-members:

``LossReserve``
^^^^^^^^^^^^^^^

.. autoclass:: gemact.lossreserve.LossReserve
   :no-members:

.. automethod:: gemact.lossreserve.LossReserve.plot_ss_fl

.. automethod:: gemact.lossreserve.LossReserve.plot_alpha_fl

.. automethod:: gemact.lossreserve.LossReserve.print_loss_reserve

.. automethod:: gemact.lossreserve.LossReserve.mean

.. automethod:: gemact.lossreserve.LossReserve.std

.. automethod:: gemact.lossreserve.LossReserve.var

.. automethod:: gemact.lossreserve.LossReserve.skewness

.. automethod:: gemact.lossreserve.LossReserve.ppf

.. automethod:: gemact.lossreserve.LossReserve.cdf

.. automethod:: gemact.lossreserve.LossReserve.sf


``LossAggregation``
-------------------

Loss aggregation
~~~~~~~~~~~~~~~~

.. math::
   :label: pty

   P\left[ X_1 +\ldots +X_d \right] \approx P_n(s)

The probability in equation :eq:`pty` can be approximated iteratively via the AEP algorithm, which is implemented in GEMAct under the following assumptions:

* :math:`X=(X_1, \ldots, X_d)` is a vector of strictly positive random components.

* The joint c.d.f. :math:`H(x_1,\ldots,x_d)=P\left[ X_1 +\ldots +X_d \right]` is known or it can be computed numerically.

Refer to :cite:t:`arbenz11` for an extensive explanation of the AEP algorithm.
It is possible to use Monte Carlo simulation for comparison.

Example
^^^^^^^

Example code under a Frank copula assumption::

    from gemact.lossaggregation import LossAggregation, Copula, Margins

    lossaggregation = LossAggregation(
        margins=Margins(
            dist=['genpareto', 'lognormal'],
            par=[
                {'loc': 0, 'scale': 1/.9, 'c': 1/.9},
                {'loc': 0, 'scale': 10, 'shape': 1.5}
            ],
        ),
        copula=Copula(
            dist='frank',
            par={'par': 1.2, 'dim': 2}
        ),
        n_sim=500000,
        random_state=10,
        n_iter=8
    )

    s = 300
    p_aep = lossaggregation.cdf(x=s, method='aep')
    p_mc = lossaggregation.cdf(x=s, method='mc')

Classes
~~~~~~~~~~~~~~

``Margins``
^^^^^^^^^^^

.. autoclass:: gemact.lossaggregation.Margins
   :no-members:

.. automethod:: gemact.lossaggregation.Margins.ppf

.. automethod:: gemact.lossaggregation.Margins.cdf

``Copula``
^^^^^^^^^^

.. autoclass:: gemact.lossaggregation.Copula
   :no-members:

.. automethod:: gemact.lossaggregation.Copula.rvs

.. automethod:: gemact.lossaggregation.Copula.cdf

``LossAggregation``
^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.lossaggregation.LossAggregation
   :no-members:

.. automethod:: gemact.lossaggregation.LossAggregation.dist_calculate

.. automethod:: gemact.lossaggregation.LossAggregation.cdf

.. automethod:: gemact.lossaggregation.LossAggregation.sf

.. automethod:: gemact.lossaggregation.LossAggregation.ppf

.. automethod:: gemact.lossaggregation.LossAggregation.rvs

.. automethod:: gemact.lossaggregation.LossAggregation.moment

.. automethod:: gemact.lossaggregation.LossAggregation.mean

.. automethod:: gemact.lossaggregation.LossAggregation.skewness

.. automethod:: gemact.lossaggregation.LossAggregation.var

.. automethod:: gemact.lossaggregation.LossAggregation.std

.. automethod:: gemact.lossaggregation.LossAggregation.lev

.. automethod:: gemact.lossaggregation.LossAggregation.censored_moment

.. automethod:: gemact.lossaggregation.LossAggregation.plot_cdf

``distributions`` module
------------------------

The distribution classes below are documented at class level and through explicitly documented methods. Inherited private base classes are intentionally not expanded here to avoid repeated private-class documentation.

Frequency distributions
~~~~~~~~~~~~~~~~~~~~~~~

``Poisson``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.Poisson
   :no-members:

.. automethod:: gemact.distributions.Poisson.pgf

.. automethod:: gemact.distributions.Poisson.par_deductible_adjuster

.. automethod:: gemact.distributions.Poisson.par_deductible_reverter

.. automethod:: gemact.distributions.Poisson.abk

.. automethod:: gemact.distributions.Poisson.skewness

.. automethod:: gemact.distributions.Poisson.kurtosis

``Binom``
^^^^^^^^^

.. autoclass:: gemact.distributions.Binom
   :no-members:

.. automethod:: gemact.distributions.Binom.pgf

.. automethod:: gemact.distributions.Binom.par_deductible_adjuster

.. automethod:: gemact.distributions.Binom.par_deductible_reverter

.. automethod:: gemact.distributions.Binom.abk

.. automethod:: gemact.distributions.Binom.skewness

.. automethod:: gemact.distributions.Binom.kurtosis

``Geom``
^^^^^^^^

.. autoclass:: gemact.distributions.Geom
   :no-members:

.. automethod:: gemact.distributions.Geom.pgf

.. automethod:: gemact.distributions.Geom.par_deductible_adjuster

.. automethod:: gemact.distributions.Geom.par_deductible_reverter

.. automethod:: gemact.distributions.Geom.abk

.. automethod:: gemact.distributions.Geom.skewness

.. automethod:: gemact.distributions.Geom.kurtosis

``NegBinom``
^^^^^^^^^^^^

.. autoclass:: gemact.distributions.NegBinom
   :no-members:

.. automethod:: gemact.distributions.NegBinom.pgf

.. automethod:: gemact.distributions.NegBinom.par_deductible_adjuster

.. automethod:: gemact.distributions.NegBinom.par_deductible_reverter

.. automethod:: gemact.distributions.NegBinom.abk

.. automethod:: gemact.distributions.NegBinom.skewness

.. automethod:: gemact.distributions.NegBinom.kurtosis

``Logser``
^^^^^^^^^^

.. autoclass:: gemact.distributions.Logser
   :no-members:

.. automethod:: gemact.distributions.Logser.pgf

.. automethod:: gemact.distributions.Logser.abk

.. automethod:: gemact.distributions.Logser.par_deductible_adjuster

.. automethod:: gemact.distributions.Logser.par_deductible_reverter

``ZTPoisson``
^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.ZTPoisson
   :no-members:

.. automethod:: gemact.distributions.ZTPoisson.pmf

.. automethod:: gemact.distributions.ZTPoisson.logpmf

.. automethod:: gemact.distributions.ZTPoisson.cdf

.. automethod:: gemact.distributions.ZTPoisson.logcdf

.. automethod:: gemact.distributions.ZTPoisson.rvs

.. automethod:: gemact.distributions.ZTPoisson.ppf

.. automethod:: gemact.distributions.ZTPoisson.mean

.. automethod:: gemact.distributions.ZTPoisson.var

.. automethod:: gemact.distributions.ZTPoisson.pgf

.. automethod:: gemact.distributions.ZTPoisson.abk

.. automethod:: gemact.distributions.ZTPoisson.par_deductible_adjuster

.. automethod:: gemact.distributions.ZTPoisson.par_deductible_reverter

``ZMPoisson``
^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.ZMPoisson
   :no-members:

.. automethod:: gemact.distributions.ZMPoisson.pmf

.. automethod:: gemact.distributions.ZMPoisson.logpmf

.. automethod:: gemact.distributions.ZMPoisson.cdf

.. automethod:: gemact.distributions.ZMPoisson.logcdf

.. automethod:: gemact.distributions.ZMPoisson.rvs

.. automethod:: gemact.distributions.ZMPoisson.ppf

.. automethod:: gemact.distributions.ZMPoisson.pgf

.. automethod:: gemact.distributions.ZMPoisson.abk

.. automethod:: gemact.distributions.ZMPoisson.par_deductible_adjuster

.. automethod:: gemact.distributions.ZMPoisson.par_deductible_reverter

``ZTBinom``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.ZTBinom
   :no-members:

.. automethod:: gemact.distributions.ZTBinom.pmf

.. automethod:: gemact.distributions.ZTBinom.logpmf

.. automethod:: gemact.distributions.ZTBinom.cdf

.. automethod:: gemact.distributions.ZTBinom.logcdf

.. automethod:: gemact.distributions.ZTBinom.rvs

.. automethod:: gemact.distributions.ZTBinom.ppf

.. automethod:: gemact.distributions.ZTBinom.mean

.. automethod:: gemact.distributions.ZTBinom.var

.. automethod:: gemact.distributions.ZTBinom.pgf

.. automethod:: gemact.distributions.ZTBinom.abk

.. automethod:: gemact.distributions.ZTBinom.par_deductible_adjuster

.. automethod:: gemact.distributions.ZTBinom.par_deductible_reverter

``ZMBinom``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.ZMBinom
   :no-members:

.. automethod:: gemact.distributions.ZMBinom.pmf

.. automethod:: gemact.distributions.ZMBinom.logpmf

.. automethod:: gemact.distributions.ZMBinom.cdf

.. automethod:: gemact.distributions.ZMBinom.logcdf

.. automethod:: gemact.distributions.ZMBinom.rvs

.. automethod:: gemact.distributions.ZMBinom.ppf

.. automethod:: gemact.distributions.ZMBinom.pgf

.. automethod:: gemact.distributions.ZMBinom.abk

.. automethod:: gemact.distributions.ZMBinom.par_deductible_adjuster

.. automethod:: gemact.distributions.ZMBinom.par_deductible_reverter

``ZTGeom``
^^^^^^^^^^

.. autoclass:: gemact.distributions.ZTGeom
   :no-members:

.. automethod:: gemact.distributions.ZTGeom.pmf

.. automethod:: gemact.distributions.ZTGeom.logpmf

.. automethod:: gemact.distributions.ZTGeom.cdf

.. automethod:: gemact.distributions.ZTGeom.logcdf

.. automethod:: gemact.distributions.ZTGeom.rvs

.. automethod:: gemact.distributions.ZTGeom.ppf

.. automethod:: gemact.distributions.ZTGeom.mean

.. automethod:: gemact.distributions.ZTGeom.var

.. automethod:: gemact.distributions.ZTGeom.pgf

.. automethod:: gemact.distributions.ZTGeom.abk

.. automethod:: gemact.distributions.ZTGeom.par_deductible_adjuster

.. automethod:: gemact.distributions.ZTGeom.par_deductible_reverter

``ZMGeom``
^^^^^^^^^^

.. autoclass:: gemact.distributions.ZMGeom
   :no-members:

.. automethod:: gemact.distributions.ZMGeom.pmf

.. automethod:: gemact.distributions.ZMGeom.logpmf

.. automethod:: gemact.distributions.ZMGeom.cdf

.. automethod:: gemact.distributions.ZMGeom.logcdf

.. automethod:: gemact.distributions.ZMGeom.rvs

.. automethod:: gemact.distributions.ZMGeom.ppf

.. automethod:: gemact.distributions.ZMGeom.pgf

.. automethod:: gemact.distributions.ZMGeom.abk

.. automethod:: gemact.distributions.ZMGeom.par_deductible_adjuster

.. automethod:: gemact.distributions.ZMGeom.par_deductible_reverter

``ZTNegBinom``
^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.ZTNegBinom
   :no-members:

.. automethod:: gemact.distributions.ZTNegBinom.pmf

.. automethod:: gemact.distributions.ZTNegBinom.logpmf

.. automethod:: gemact.distributions.ZTNegBinom.cdf

.. automethod:: gemact.distributions.ZTNegBinom.logcdf

.. automethod:: gemact.distributions.ZTNegBinom.rvs

.. automethod:: gemact.distributions.ZTNegBinom.ppf

.. automethod:: gemact.distributions.ZTNegBinom.mean

.. automethod:: gemact.distributions.ZTNegBinom.var

.. automethod:: gemact.distributions.ZTNegBinom.pgf

.. automethod:: gemact.distributions.ZTNegBinom.abk

.. automethod:: gemact.distributions.ZTNegBinom.par_deductible_adjuster

.. automethod:: gemact.distributions.ZTNegBinom.par_deductible_reverter

``ZMNegBinom``
^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.ZMNegBinom
   :no-members:

.. automethod:: gemact.distributions.ZMNegBinom.pmf

.. automethod:: gemact.distributions.ZMNegBinom.logpmf

.. automethod:: gemact.distributions.ZMNegBinom.cdf

.. automethod:: gemact.distributions.ZMNegBinom.logcdf

.. automethod:: gemact.distributions.ZMNegBinom.rvs

.. automethod:: gemact.distributions.ZMNegBinom.ppf

.. automethod:: gemact.distributions.ZMNegBinom.pgf

.. automethod:: gemact.distributions.ZMNegBinom.abk

.. automethod:: gemact.distributions.ZMNegBinom.par_deductible_adjuster

.. automethod:: gemact.distributions.ZMNegBinom.par_deductible_reverter

``ZMLogser``
^^^^^^^^^^^^

.. autoclass:: gemact.distributions.ZMLogser
   :no-members:

.. automethod:: gemact.distributions.ZMLogser.pmf

.. automethod:: gemact.distributions.ZMLogser.logpmf

.. automethod:: gemact.distributions.ZMLogser.cdf

.. automethod:: gemact.distributions.ZMLogser.logcdf

.. automethod:: gemact.distributions.ZMLogser.rvs

.. automethod:: gemact.distributions.ZMLogser.ppf

.. automethod:: gemact.distributions.ZMLogser.abk

.. automethod:: gemact.distributions.ZMLogser.par_deductible_adjuster

.. automethod:: gemact.distributions.ZMLogser.par_deductible_reverter

Severity distributions
~~~~~~~~~~~~~~~~~~~~~~

``Beta``
^^^^^^^^

.. autoclass:: gemact.distributions.Beta
   :no-members:

.. automethod:: gemact.distributions.Beta.lev

``Exponential``
^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.Exponential
   :no-members:

.. automethod:: gemact.distributions.Exponential.pdf

.. automethod:: gemact.distributions.Exponential.logpdf

.. automethod:: gemact.distributions.Exponential.cdf

.. automethod:: gemact.distributions.Exponential.logcdf

.. automethod:: gemact.distributions.Exponential.sf

.. automethod:: gemact.distributions.Exponential.logsf

.. automethod:: gemact.distributions.Exponential.isf

.. automethod:: gemact.distributions.Exponential.rvs

.. automethod:: gemact.distributions.Exponential.entropy

.. automethod:: gemact.distributions.Exponential.mean

.. automethod:: gemact.distributions.Exponential.var

.. automethod:: gemact.distributions.Exponential.std

.. automethod:: gemact.distributions.Exponential.ppf

.. automethod:: gemact.distributions.Exponential.lev

.. automethod:: gemact.distributions.Exponential.partial_moment

``Gamma``
^^^^^^^^^

.. autoclass:: gemact.distributions.Gamma
   :no-members:

.. automethod:: gemact.distributions.Gamma.lev

.. automethod:: gemact.distributions.Gamma.partial_moment

``InvGamma``
^^^^^^^^^^^^

.. autoclass:: gemact.distributions.InvGamma
   :no-members:

.. automethod:: gemact.distributions.InvGamma.lev

``GenPareto``
^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.GenPareto
   :no-members:

.. automethod:: gemact.distributions.GenPareto.lev

``Pareto2``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.Pareto2
   :no-members:

``Pareto1``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.Pareto1
   :no-members:

``Lognormal``
^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.Lognormal
   :no-members:

.. automethod:: gemact.distributions.Lognormal.lev

.. automethod:: gemact.distributions.Lognormal.partial_moment

``GenBeta``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.GenBeta
   :no-members:

.. automethod:: gemact.distributions.GenBeta.rvs

.. automethod:: gemact.distributions.GenBeta.pdf

.. automethod:: gemact.distributions.GenBeta.cdf

.. automethod:: gemact.distributions.GenBeta.logpdf

.. automethod:: gemact.distributions.GenBeta.logcdf

.. automethod:: gemact.distributions.GenBeta.sf

.. automethod:: gemact.distributions.GenBeta.logsf

.. automethod:: gemact.distributions.GenBeta.ppf

.. automethod:: gemact.distributions.GenBeta.isf

.. automethod:: gemact.distributions.GenBeta.moment

.. automethod:: gemact.distributions.GenBeta.stats

.. automethod:: gemact.distributions.GenBeta.median

.. automethod:: gemact.distributions.GenBeta.mean

.. automethod:: gemact.distributions.GenBeta.var

.. automethod:: gemact.distributions.GenBeta.std

.. automethod:: gemact.distributions.GenBeta.skewness

.. automethod:: gemact.distributions.GenBeta.kurtosis

.. automethod:: gemact.distributions.GenBeta.lev

.. automethod:: gemact.distributions.GenBeta.censored_moment

.. automethod:: gemact.distributions.GenBeta.partial_moment

.. automethod:: gemact.distributions.GenBeta.truncated_moment

``Burr12``
^^^^^^^^^^

.. autoclass:: gemact.distributions.Burr12
   :no-members:

.. automethod:: gemact.distributions.Burr12.lev

``Paralogistic``
^^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.Paralogistic
   :no-members:

``Dagum``
^^^^^^^^^

.. autoclass:: gemact.distributions.Dagum
   :no-members:

.. automethod:: gemact.distributions.Dagum.lev

``InvParalogistic``
^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.InvParalogistic
   :no-members:

``Weibull``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.Weibull
   :no-members:

.. automethod:: gemact.distributions.Weibull.lev

``InvWeibull``
^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.InvWeibull
   :no-members:

.. automethod:: gemact.distributions.InvWeibull.lev

``InvGauss``
^^^^^^^^^^^^

.. autoclass:: gemact.distributions.InvGauss
   :no-members:

.. automethod:: gemact.distributions.InvGauss.lev

``Fisk``
^^^^^^^^

.. autoclass:: gemact.distributions.Fisk
   :no-members:

.. automethod:: gemact.distributions.Fisk.lev

``LogGamma``
^^^^^^^^^^^^

.. autoclass:: gemact.distributions.LogGamma
   :no-members:

.. automethod:: gemact.distributions.LogGamma.pdf

.. automethod:: gemact.distributions.LogGamma.cdf

.. automethod:: gemact.distributions.LogGamma.sf

.. automethod:: gemact.distributions.LogGamma.ppf

.. automethod:: gemact.distributions.LogGamma.rvs

.. automethod:: gemact.distributions.LogGamma.moment

.. automethod:: gemact.distributions.LogGamma.mean

.. automethod:: gemact.distributions.LogGamma.var

.. automethod:: gemact.distributions.LogGamma.std

.. automethod:: gemact.distributions.LogGamma.logpdf

.. automethod:: gemact.distributions.LogGamma.logcdf

.. automethod:: gemact.distributions.LogGamma.logsf

.. automethod:: gemact.distributions.LogGamma.isf

.. automethod:: gemact.distributions.LogGamma.stats

.. automethod:: gemact.distributions.LogGamma.median

.. automethod:: gemact.distributions.LogGamma.skewness

.. automethod:: gemact.distributions.LogGamma.kurtosis

.. automethod:: gemact.distributions.LogGamma.lev

.. automethod:: gemact.distributions.LogGamma.censored_moment

.. automethod:: gemact.distributions.LogGamma.partial_moment

.. automethod:: gemact.distributions.LogGamma.truncated_moment

``Uniform``
^^^^^^^^^^^

.. autoclass:: gemact.distributions.Uniform
   :no-members:

Empirical and piecewise distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PWL``
^^^^^^^

.. autoclass:: gemact.distributions.PWL
   :no-members:

.. automethod:: gemact.distributions.PWL.cdf

.. automethod:: gemact.distributions.PWL.pdf

.. automethod:: gemact.distributions.PWL.ppf

.. automethod:: gemact.distributions.PWL.rvs

.. automethod:: gemact.distributions.PWL.sf

.. automethod:: gemact.distributions.PWL.moment

.. automethod:: gemact.distributions.PWL.mean

.. automethod:: gemact.distributions.PWL.var

.. automethod:: gemact.distributions.PWL.std

.. automethod:: gemact.distributions.PWL.skewness

.. automethod:: gemact.distributions.PWL.kurtosis

.. automethod:: gemact.distributions.PWL.censored_moment

.. automethod:: gemact.distributions.PWL.lev

``PWC``
^^^^^^^

.. autoclass:: gemact.distributions.PWC
   :no-members:

.. automethod:: gemact.distributions.PWC.cdf

.. automethod:: gemact.distributions.PWC.ppf

.. automethod:: gemact.distributions.PWC.rvs

.. automethod:: gemact.distributions.PWC.sf

.. automethod:: gemact.distributions.PWC.moment

.. automethod:: gemact.distributions.PWC.mean

.. automethod:: gemact.distributions.PWC.var

.. automethod:: gemact.distributions.PWC.std

.. automethod:: gemact.distributions.PWC.skewness

.. automethod:: gemact.distributions.PWC.kurtosis

.. automethod:: gemact.distributions.PWC.censored_moment

.. automethod:: gemact.distributions.PWC.lev

Multivariate distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

``Multinomial``
^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.Multinomial
   :no-members:

.. automethod:: gemact.distributions.Multinomial.cov

.. automethod:: gemact.distributions.Multinomial.var

.. automethod:: gemact.distributions.Multinomial.entropy

.. automethod:: gemact.distributions.Multinomial.pmf

.. automethod:: gemact.distributions.Multinomial.logpmf

``Dirichlet_Multinomial``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.Dirichlet_Multinomial
   :no-members:

.. automethod:: gemact.distributions.Dirichlet_Multinomial.cov

.. automethod:: gemact.distributions.Dirichlet_Multinomial.var

.. automethod:: gemact.distributions.Dirichlet_Multinomial.pmf

.. automethod:: gemact.distributions.Dirichlet_Multinomial.logpmf

.. automethod:: gemact.distributions.Dirichlet_Multinomial.mean

.. automethod:: gemact.distributions.Dirichlet_Multinomial.rvs

``NegMultinom``
^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.NegMultinom
   :no-members:

.. automethod:: gemact.distributions.NegMultinom.pmf

.. automethod:: gemact.distributions.NegMultinom.logpmf

.. automethod:: gemact.distributions.NegMultinom.mean

.. automethod:: gemact.distributions.NegMultinom.var

.. automethod:: gemact.distributions.NegMultinom.cov

.. automethod:: gemact.distributions.NegMultinom.rvs

.. automethod:: gemact.distributions.NegMultinom.mgf

``MultivariateBinomial``
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.MultivariateBinomial
   :no-members:

.. automethod:: gemact.distributions.MultivariateBinomial.pmf

.. automethod:: gemact.distributions.MultivariateBinomial.mean

.. automethod:: gemact.distributions.MultivariateBinomial.cov

.. automethod:: gemact.distributions.MultivariateBinomial.var

.. automethod:: gemact.distributions.MultivariateBinomial.correlation

.. automethod:: gemact.distributions.MultivariateBinomial.rvs

``MultivariatePoisson``
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.distributions.MultivariatePoisson
   :no-members:

.. automethod:: gemact.distributions.MultivariatePoisson.pmf

.. automethod:: gemact.distributions.MultivariatePoisson.mean

.. automethod:: gemact.distributions.MultivariatePoisson.cov

.. automethod:: gemact.distributions.MultivariatePoisson.var

.. automethod:: gemact.distributions.MultivariatePoisson.correlation

.. automethod:: gemact.distributions.MultivariatePoisson.rvs

``copulas`` module
------------------

``ClaytonCopula``
^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.ClaytonCopula
   :no-members:

.. automethod:: gemact.copulas.ClaytonCopula.cdf

.. automethod:: gemact.copulas.ClaytonCopula.rvs

``FrankCopula``
^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.FrankCopula
   :no-members:

.. automethod:: gemact.copulas.FrankCopula.cdf

.. automethod:: gemact.copulas.FrankCopula.rvs

``GumbelCopula``
^^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.GumbelCopula
   :no-members:

.. automethod:: gemact.copulas.GumbelCopula.cdf

.. automethod:: gemact.copulas.GumbelCopula.rvs

``GaussCopula``
^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.GaussCopula
   :no-members:

.. automethod:: gemact.copulas.GaussCopula.cdf

.. automethod:: gemact.copulas.GaussCopula.rvs

``TCopula``
^^^^^^^^^^^

.. autoclass:: gemact.copulas.TCopula
   :no-members:

.. automethod:: gemact.copulas.TCopula.cdf

.. automethod:: gemact.copulas.TCopula.rvs

``IndependenceCopula``
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.IndependenceCopula
   :no-members:

.. automethod:: gemact.copulas.IndependenceCopula.cdf

.. automethod:: gemact.copulas.IndependenceCopula.rvs

``FHLowerCopula``
^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.FHLowerCopula
   :no-members:

.. automethod:: gemact.copulas.FHLowerCopula.cdf

.. automethod:: gemact.copulas.FHLowerCopula.rvs

``FHUpperCopula``
^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.FHUpperCopula
   :no-members:

.. automethod:: gemact.copulas.FHUpperCopula.cdf

.. automethod:: gemact.copulas.FHUpperCopula.rvs

``JoeCopula``
^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.JoeCopula
   :no-members:

.. automethod:: gemact.copulas.JoeCopula.cdf

.. automethod:: gemact.copulas.JoeCopula.rvs

``AliMikhailHaqCopula``
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gemact.copulas.AliMikhailHaqCopula
   :no-members:

.. automethod:: gemact.copulas.AliMikhailHaqCopula.cdf

.. automethod:: gemact.copulas.AliMikhailHaqCopula.rvs

``helperfunctions`` module
--------------------------

The following helper functions have docstrings. They are listed explicitly and private helpers are not included.

``arg_type_handler``
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.arg_type_handler

``ecdf``
^^^^^^^^

.. autofunction:: gemact.helperfunctions.ecdf

``normalizernans``
^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.normalizernans

``lrcrm_f1``
^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.lrcrm_f1

``lrcrm_f2``
^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.lrcrm_f2

``lrcrm_f3``
^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.lrcrm_f3

``cartesian_product``
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.cartesian_product

``cov_to_corr``
^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.cov_to_corr

``multivariate_t_cdf``
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.multivariate_t_cdf

``assert_member``
^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.assert_member

``assert_type_value``
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.assert_type_value

``ndarray_try_convert``
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.ndarray_try_convert

``check_condition``
^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.check_condition

``handle_random_state``
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.handle_random_state

``assert_not_none``
^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.assert_not_none

``check_none``
^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.check_none

``layerFunc``
^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.layerFunc

``triangle_dimension``
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.triangle_dimension

``compute_block2_crm_msep``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.compute_block2_crm_msep

``lrcrm_skewness_f4``
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.lrcrm_skewness_f4

``compute_block2_crm_skewness``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.compute_block2_crm_skewness

``compute_block3_crm_skewness``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.compute_block3_crm_skewness

``find_diagonal``
^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.find_diagonal

``incrementals_2_cumulatives``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.incrementals_2_cumulatives

``make_pdf``
^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.make_pdf

``make_cdf``
^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.make_cdf

``find_interval``
^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.find_interval

``simulate``
^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.simulate

``memoize``
^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.memoize

``partial_moment``
^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.partial_moment

``censored_moment``
^^^^^^^^^^^^^^^^^^^

.. autofunction:: gemact.helperfunctions.censored_moment


References
----------

.. bibliography:: refs.bib
