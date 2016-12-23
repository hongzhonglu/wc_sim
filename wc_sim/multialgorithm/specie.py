'''Track the population of a single specie in a multi-algorithmic model.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-07-19
:Copyright: 2016, Karr Lab
:License: MIT
'''

from wc_sim.multialgorithm.config import paths as config_paths
from wc_utils.config.core import ConfigManager
from wc_utils.util.rand import RandomStateManager
from wc_sim.multialgorithm.multialgorithm_errors import NegativePopulationError

config = ConfigManager(config_paths.core).get_config()['wc_sim']['multialgorithm']


class Specie(object):
    '''Specie tracks the population of a single specie in a multi-algorithmic model.

    A specie is a shared object that can be read and written by multiple submodels in a
    multi-algorithmic model. We assume that a sequence of accesses of a specie instance will
    occur in non-decreasing simulation time order. (This assumption holds for conservative discrete
    event simulations and all committed parts of optimistic parallel simulations like Time Warp.)

    Consider a multi-algorithmic model that contains both discrete-time submodels, like the
    stochastic simulation algorithm (SSA), and continuous-time submodels, like ODEs and FBA.
    Discrete-time algorithms change system state at discrete time instants. Continuous-time
    algorithms employ continuous models of state change, and sample these models at time instants
    determined by the algorithm. At these instants, continuous-time models typically
    estimate a specie's population and the population's rate of change. We assume this behavior.

    A specie's state in a multi-algorithmic model may be modeled by multiple submodels which model
    reactions in which the specie participates. These can be multiple discrete-time submodels and
    at most one continuous-time submodel. (If multiple continuous-time submodels were allowed to
    predict reactions that involve a specie, a mechanism would be needed to reconsile conflicting
    flux values. We have not addressed that issue yet.)

    Discrete-time and continuous-time models adjust the state of a species by the methods
    `discrete_adjustment()` and `continuous_adjustment()`, respectively. These adjustments take the
    following forms,

    * `discrete_adjustment( population_change, time )`
    * `continuous_adjustment( population_change, time, flux )`

    where `population_change` is the increase or decrease in the specie's population, `time` is the
    time at which that change takes place, and `flux` is the predicted future rate of change of the
    population.

    To improve the accuracy of multi-algorithmic models, we support linear *interpolation* of
    population predictions for species modeled by a continuous-time submodel. An interpolated
    prediction is based on the most recent continuous-time flux prediction. Thus, we assume
    that a population modeled by a continuous model is adjusted sufficiently frequently
    that the most recent adjustment accurately estimates flux.

    A specie instance stores the most recent value of the specie's population in `last_population`,
    which is initialized when the instance is created. If a specie is modeled by a
    continuous-time submodel, it also stores the specie's flux in `continuous_flux` and the time
    of the most recent `continuous_adjustment` in `continuous_time`. Otherwise, `continuous_time`
    is `None`. Interpolation determines the population prediction `p` at time `t` as::

        interpolation = 0
        if not continuous_time is None:
            interpolation = (t - continuous_time)*continuous_flux
        p = last_population + interpolation

    This approach is completely general, and can be applied to any simulation value
    whose dynamics are predicted by a multi-algorithmic model.

    Population values returned by specie's methods use stochastic rounding to provide integer
    values and avoid systematic rounding bias. See more detail in `get_population`'s docstring.

    Attributes:
        specie_name (str): the specie's name; not logically needed, but helpful for error
            reporting, logging, debugging, etc.
        last_population (float): population after the most recent adjustment
        continuous_submodel (bool): whether one of the submodels modeling the species is a
            continuous submodel; must be set at initialization
        continuous_flux (float): if a continuous submodel is modeling the specie, the flux provided
            at initialization or by the most recent adjustment by a continuous model
        continuous_time (float): if a continuous submodel is modeling the specie, the simulation
            time of initialization (0) or the most recent adjustment by the continuous model

    '''
    # use __slots__ to save space
    __slots__ = ['specie_name', 'last_population', 'continuous_time', 'continuous_flux',
        'random_state', 'continuous_submodel']

    def __init__(self, specie_name, initial_population, initial_flux=None):
        '''Initialize a specie object at simulation time 0.

        Args:
            specie_name (str): the specie's name; not logically needed, but helpful for error
                reporting, logging, debugging, etc.
            initial_population (int): non-negative number; initial population of the specie
            initial_flux (number, optional): initial flux for the specie; required for species whose
                population is estimated, at least in part, by a continuous model
        '''
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.specie_name = specie_name
        self.last_population = initial_population
        self.continuous_submodel = False
        if initial_flux is not None:
            self.continuous_submodel = True
            self.continuous_time = 0
            self.continuous_flux = initial_flux

        self.random_state = RandomStateManager.instance()

    def discrete_adjustment(self, population_change, time):
        '''Make a discrete adjustment of the specie's population.

        A discrete-time submodel, such as the stochastic simulation algorithm (SSA), must use this
        method to adjust the specie's population.

        Args:
            population_change (number): the modeled increase or decrease in the specie's population
            time (number): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between integrations.

        Returns:
            int: an integer approximation of the specie's adjusted population

        Raises:
            NegativePopulationError: if the predicted population at `time` is negative or
            if decreasing the population by `population_change` would make the population negative
        '''
        current_population = self.get_population(time)
        if current_population + population_change < 0:
            raise NegativePopulationError('discrete_adjustment', self.specie_name,
                self.last_population, population_change)
        self.last_population += population_change
        return self.get_population(time)

    def continuous_adjustment(self, population_change, time, flux):
        '''A continuous-time submodel adjusts the specie's state.

        A continuous-time submodel, such as an ordinary differential equation (ODE) or a dynamic flux
        balance analysis (FBA) model, uses this method to adjust the specie's state. Each
        integration of a continuous-time model must predict a specie's population change and the
        population's short-term future rate of change, i.e., its `flux`. Further, since an
        integration of a continuous-time model at the current time must depend on this specie's
        population just before the integration, we assume that `population_change` incorporates
        population changes predicted by the flux provided by the previous `continuous_adjustment`
        call.

        Args:
            population_change (number): modeled increase or decrease in the specie's population
            time (number): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between
                integrations.
            flux (number): the predicted flux of the specie at the provided time

        Returns:
            int: the specie's adjusted population, rounded to an integer

        Raises:
            ValueError: if `time` is not greater than the time of the most recent
                `continuous_adjustment` call on this `specie`
            NegativePopulationError: if applying `population_change` makes the population go negative
        '''
        if not self.continuous_submodel:
            raise ValueError("continuous_adjustment(): initial flux was not provided")
        # the simulation time must advance between adjacent continuous adjustments
        if time <= self.continuous_time:
            raise ValueError("continuous_adjustment(): time <= self.continuous_time: "
                "{:.2f} < {:.2f}".format(time, self.continuous_time))
        if self.last_population + population_change < 0:
            raise NegativePopulationError('continuous_adjustment', self.specie_name,
                self.last_population, population_change, time-self.continuous_time)
        self.continuous_time = time
        self.continuous_flux = flux
        self.last_population += population_change
        return self.get_population(time)

    def get_population(self, time=None):
        '''Provide the specie's current population.

        If one of the submodel(s) predicting the specie's population is a continuous-time model,
        then use the specie's last flux to interpolate the current population, as described in the
        class documentation.

        Clearly, species populations in biological systems are non-negative integers. However,
        continuous-time models approximate populations with continuous representations, and
        therefore predict real, non-integral, populations. But discrete-time models like SSA
        do not naturally handle non-integral copy numbers.

        We resolve this conflict by storing real valued populations within a specie, but
        providing only integral population predictions. To aovid the bias that would arise by
        always using floor() or ceiling() to convert a float to an integer, population predictions
        are stochastically rounded before being returned by `get_population`. *This means
        that a sequence of calls to `get_population` which do not have any interveening
        adjustment operations may **NOT** return a sequence of equal population values.*

        Args:
            time (number, optional): the current simulation time; `time` is required if one of the
                submodels modeling the specie is a continuous-time submodel.

        Returns:
            int: an integer approximation of the specie's adjusted population

        Raises:
            ValueError: if `time` is required but not provided
            ValueError: if `time` is earlier than the time of a previous continuous adjustment
            NegativePopulationError: if interpolation predicts a negative population
        '''
        if not self.continuous_submodel:
            return self.random_state.round( self.last_population )
        else:
            if time is None:
                raise ValueError("get_population(): time needed because "
                    "continuous adjustment received at time {:.2f}".format(self.continuous_time))
            if time < self.continuous_time:
                raise ValueError("get_population(): time < self.continuous_time: {:.2f} < {:.2f}\n".format(
                    time, self.continuous_time))
            interpolation=0
            # TODO(Arthur): compare with and wo interpolation
            if config['interpolate']:
                interpolation = (time - self.continuous_time) * self.continuous_flux
            if self.last_population + interpolation < 0:
                raise NegativePopulationError('get_population', self.specie_name,
                    self.last_population, interpolation, time - self.continuous_time)
            float_copy_number = self.last_population + interpolation
            return self.random_state.round( float_copy_number )

    def __str__(self):
        if self.continuous_submodel:
            return "specie_name: {}; last_population: {}; continuous_time: {}; continuous_flux: {}".format(
                self.specie_name, self.last_population, self.continuous_time, self.continuous_flux)
        else:
            return "specie_name: {}; last_population: {}".format(
                self.specie_name, self.last_population)

    @staticmethod
    def heading():
        '''Return a heading for a tab-separated table of species data.'''
        return '\t'.join('specie_name last_population continuous_time continuous_flux'.split())

    def row(self):
        '''Return a row for a tab-separated table of species data.'''
        if self.continuous_submodel:
            return "{}\t{:.2f}\t{:.2f}\t{:.2f}".format(self.specie_name, self.last_population, self.continuous_time, self.continuous_flux)
            '\t'.join([])
        else:
            return "{}\t{:.2f}".format(self.specie_name, self.last_population)
            '\t'.join([])
