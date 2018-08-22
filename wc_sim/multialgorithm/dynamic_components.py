""" Dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-02-07
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import numpy as np
import math
import numbers
from enum import Enum
import warnings

from obj_model import utils
from wc_lang.core import Species, SpeciesType, Compartment
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.dynamic_expressions import (DynamicSpecies, DynamicFunction, DynamicStopCondition,
    DynamicParameter, DynamicObservable)


class DynamicCompartmentType(Enum):
    """ Types of dynamic compartments """
    # Represent physical biochemistry: species have mass, density is constant, volume=mass/density is always computable
    biochemical = 1
    # Represent abstract species: constant volume and, perhaps, an unknown density
    abstract = 2


class DynamicCompartment(object):
    """ A dynamic compartment

    A `DynamicCompartment` tracks the dynamic aggregate state of a compartment, primarily its
    mass and volume. A `DynamicCompartment` is created for each `wc_lang` `Compartment` in a whole-cell
    model.

    Attributes:
        id (:obj:`str`): id of this `DynamicCompartment`, copied from `compartment`
        name (:obj:`str`): name of this `DynamicCompartment`, copied from `compartment`
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this `DynamicCompartment`
        species_ids (:obj:`list` of `str`): the IDs of the species stored
            in this dynamic compartment; if `None`, use the IDs of all species in `species_population`
        compartment_type (:obj:`DynamicCompartmentType`): the type of dynamic compartment
        init_volume (:obj:`float`): initial volume specified in the `wc_lang` model; if `compartment_type` is
            `DynamicCompartmentType.abstract`, the volume of this compartment remains constant
        constant_density (:obj:`float`): if `compartment_type` is `DynamicCompartmentType.biochemical`,
            the density of this compartment, which remains constant
    """
    def __init__(self, compartment, species_population, species_ids=None,
        compartment_type=DynamicCompartmentType.biochemical, density=None):
        """ Initialize this `DynamicCompartment`

        Args:
            compartment (:obj:`Compartment`): the corresponding static `wc_lang` `Compartment`
            species_population (:obj:`LocalSpeciesPopulation`): an object that represents
                the populations of species in this `DynamicCompartment`
            species_ids (:obj:`list` of `str`, optional): the IDs of the species stored
                in this compartment; defaults to the IDs of all species in `species_population`
            compartment_type (:obj:`DynamicCompartmentType`): the type of dynamic compartment; defaults
                to `DynamicCompartmentType.biochemical`; a `DynamicCompartmentType.biochemical`
                compartment must be initialized with 0<density, provided either directly or via
                `compartment.initial_volume` and mass
            density (:obj:`float`, optional): if provided, the density of this compartment, which
                is assumed constant; not necessary if the volume and mass of the compartment are
                both positive, or if `compartment_type` is `DynamicCompartmentType.abstract`

        Raises:
            :obj:`MultialgorithmError`:
                if `compartment_type` is `DynamicCompartmentType.biochemical`, an exception is raised
                    if a positive density cannot be computed from initial volume and mass, and `density`
                    is not provided, or the molecular weight of any species type in `species_population`
                    is not a positive real;
                if `compartment_type` is `DynamicCompartmentType.abstract`, an exception is raised
                    if `compartment`'s initial volume is not a positive real
        """
        self.id = compartment.id
        self.name = compartment.name
        self.init_volume = compartment.initial_volume
        self.species_population = species_population
        self.species_ids = species_ids
        self.compartment_type = compartment_type

        if compartment_type == DynamicCompartmentType.biochemical:
            # all species types must have positive molecular weights
            if species_population.invalid_weights():
                raise MultialgorithmError("DynamicCompartment '{}': in a {} dynamic compartment all species "
                    "types must have positive molecular weights, but these species do not: '{}'".format(
                    self.name, compartment_type.name, species_population.invalid_weights()))

            # compartment must be initialized with 0<density, provided either directly or via volume and mass
            if density is None:
                if not isinstance(self.init_volume, numbers.Real) or self.init_volume<=0 or math.isnan(self.init_volume):
                    raise MultialgorithmError("DynamicCompartment '{}': in a {} dynamic compartment init_volume "
                        "must be a positive real number but it is '{}'".format(
                        self.name, compartment_type.name, self.init_volume))
                if not isinstance(self.mass(), numbers.Real) or self.mass()<=0 or math.isnan(self.mass()):
                    raise MultialgorithmError("DynamicCompartment '{}': in a {} dynamic compartment initial mass "
                        "must be a positive real number but it is '{}'".format(
                        self.name, compartment_type.name, self.mass()))
                self.constant_density = self.mass()/self.init_volume
            else:
                if not isinstance(density, numbers.Real) or density<=0 or math.isnan(density):
                    raise MultialgorithmError("DynamicCompartment '{}': in a {} dynamic compartment density, if "
                        "provided, must be a positive real number but it is '{}'".format(
                        self.name, compartment_type.name, density))
                if 0<self.mass():
                    warnings.warn("DynamicCompartment '{}': in a {} dynamic compartment providing density "
                        "when 0<self.mass() may cause unexpected behavior because density may not = mass/volume".format(
                        self.name, compartment_type.name))
                self.constant_density = density
        elif compartment_type == DynamicCompartmentType.abstract:
                if not isinstance(self.init_volume, numbers.Real) or self.init_volume<=0 or math.isnan(self.init_volume):
                    raise MultialgorithmError("DynamicCompartment '{}': in an {} dynamic compartment init_volume "
                        "must be a positive real number but it is '{}'".format(
                        self.name, compartment_type.name, self.init_volume))
        else:
            assert False, "DynamicCompartment '{}': invalid compartment_type: '{}'".format(self.name, compartment_type)

    def mass(self):
        """ Provide the total current mass of all species in this `DynamicCompartment`

        If `compartment_type` is `DynamicCompartmentType.biochemical`, the mass is the sum of the
        masses of all species. If `compartment_type` is `DynamicCompartmentType.abstract`, the mass
        may not be known, and `None` is returned.

        Returns:
            :obj:`float`: this compartment's total current mass (g)
        """
        if self.compartment_type == DynamicCompartmentType.biochemical:
            return self.species_population.compartmental_mass(self.id)
        else:
            return None

    def volume(self):
        """ Provide the current volume of this `DynamicCompartment`

        If `compartment_type` is `DynamicCompartmentType.biochemical`, the volume is mass/density.
        If `compartment_type` is `DynamicCompartmentType.abstract`, the volume is constant, and
        given by `self.init_volume`.

        Returns:
            :obj:`float`: this compartment's current volume (L)
        """
        if self.compartment_type == DynamicCompartmentType.biochemical:
            return self.mass()/self.constant_density
        else:
            return self.init_volume

    def density(self):
        """ Provide the density of this `DynamicCompartment`, which is assumed to be constant

        If `compartment_type` is `DynamicCompartmentType.biochemical`, the density is `constant_density`.
        If `compartment_type` is `DynamicCompartmentType.abstract`, the density is not known, and
        `None` is returned.

        Returns:
            :obj:`float`: this compartment's density (g/L)
        """
        if self.compartment_type == DynamicCompartmentType.biochemical:
            return self.constant_density
        else:
            return None

    def __str__(self):
        """ Provide a string representation of this `DynamicCompartment`

        Returns:
            :obj:`str`: a string representation of this compartment
        """
        values = []
        values.append("ID: " + self.id)
        values.append("Name: " + self.name)
        values.append("Initial volume (L): {}".format(self.init_volume))
        values.append("Compartment type: {}".format(self.compartment_type.name))
        if self.compartment_type == DynamicCompartmentType.biochemical:
            values.append("Constant density (g/L): {}".format(self.constant_density))
            values.append("Current mass (g): {}".format(self.mass()))
            values.append("Current volume (L): {}".format(self.volume()))
            values.append("Fold change volume: {}".format(self.volume()/self.init_volume))
        return "DynamicCompartment:\n{}".format('\n'.join(values))

# TODO(Arthur): define these in config data, which may come from wc_lang
EXTRACELLULAR_COMPARTMENT_ID = 'e'
WATER_ID = 'H2O'


class DynamicModel(object):
    """ Represent and access the dynamics of a whole-cell model simulation

    A `DynamicModel` provides access to dynamical components of the simulation, and
    determines aggregate properties that are not provided
    by other, more specific, dynamical components like species populations, submodels, and
    dynamic compartments.

    Attributes:
        dynamic_compartments (:obj: `dict`): map from compartment ID to `DynamicCompartment`; the simulation's
            `DynamicCompartment`s, one for each compartment in `model`
        cellular_dyn_compartments (:obj:`list`): list of the cellular compartments
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this `DynamicCompartment`
        dynamic_species (:obj:`dict` of `DynamicSpecies`): the simulation's dynamic species,
            indexed by their ids
        dynamic_observables (:obj:`dict` of `DynamicObservable`): the simulation's dynamic observables,
            indexed by their ids
        dynamic_functions (:obj:`dict` of `DynamicFunction`): the simulation's dynamic functions,
            indexed by their ids
        dynamic_stop_conditions (:obj:`dict` of `DynamicStopCondition`): the simulation's stop conditions,
            indexed by their ids
        dynamic_parameters (:obj:`dict` of `DynamicParameter`): the simulation's parameters,
            indexed by their ids
        fraction_dry_weight (:obj:`float`): fraction of the cell's weight which is not water
            a constant
        water_in_model (:obj:`bool`): if set, the model represents water
    """
    def __init__(self, model, species_population, dynamic_compartments):
        """ Prepare a `DynamicModel` for a discrete-event simulation

        Args:
            model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
            species_population (:obj:`LocalSpeciesPopulation`): an object that represents
                the populations of species in this `DynamicCompartment`
            dynamic_compartments (:obj: `dict`): the simulation's `DynamicCompartment`s, one for each
                compartment in `model`
        """
        self.dynamic_compartments = dynamic_compartments
        self.species_population = species_population
        self.num_submodels = len(model.get_submodels())

        # Classify compartments into extracellular and cellular; those which are not extracellular are cellular
        # Assumes at most one extracellular compartment
        extracellular_compartment = utils.get_component_by_id(model.get_compartments(),
            EXTRACELLULAR_COMPARTMENT_ID)

        self.cellular_dyn_compartments = []
        for dynamic_compartment in dynamic_compartments.values():
            if dynamic_compartment.id == EXTRACELLULAR_COMPARTMENT_ID:
                continue
            self.cellular_dyn_compartments.append( dynamic_compartment)

        # Does the model represent water?
        self.water_in_model = True
        for compartment in model.get_compartments():
            water_in_compartment_id = Species.gen_id(WATER_ID, compartment.id)
            if water_in_compartment_id not in [s.id() for s in compartment.species]:
                self.water_in_model = False
                break

        # cell dry weight
        self.fraction_dry_weight = utils.get_component_by_id(model.get_parameters(),
            'fractionDryWeight').value

        # === create dynamic objects that are not expressions ===
        # create dynamic parameters
        self.dynamic_parameters = {}
        for parameter in model.parameters:
            self.dynamic_parameters[parameter.id] = DynamicParameter(self, self.species_population,
                parameter, parameter.value)

        # create dynamic species
        self.dynamic_species = {}
        for species in model.get_species():
            self.dynamic_species[species.get_id()] = DynamicSpecies(self, self.species_population,
                species)

        # === create dynamic expressions ===
        # create dynamic observables
        self.dynamic_observables = {}
        for observable in model.observables:
            self.dynamic_observables[observable.id] = DynamicObservable(self, self.species_population, observable,
                observable.expression.analyzed_expr)

        # create dynamic functions
        self.dynamic_functions = {}
        for function in model.functions:
            self.dynamic_functions[function.id] = DynamicFunction(self, self.species_population, function,
                function.expression.analyzed_expr)

        # create dynamic stop conditions
        self.dynamic_stop_conditions = {}
        for stop_condition in model.stop_conditions:
            self.dynamic_stop_conditions[stop_condition.id] = DynamicStopCondition(self, self.species_population,
                stop_condition, stop_condition.expression.analyzed_expr)

        # prepare dynamic expressions
        for dynamic_expression_group in [self.dynamic_observables, self.dynamic_functions,
            self.dynamic_stop_conditions]:
            for dynamic_expression in dynamic_expression_group.values():
                dynamic_expression.prepare()

    def cell_mass(self):
        """ Compute the cell's mass

        Sum the mass of all `DynamicCompartment`s that are not extracellular.
        Assumes compartment volumes are in L and concentrations in mol/L.

        Returns:
            :obj:`float`: the cell's mass (g)
        """
        # TODO(Arthur): how should water be treated in mass calculations?
        return sum([dynamic_compartment.mass() for dynamic_compartment in self.cellular_dyn_compartments])

    def cell_volume(self):
        """ Compute the cell's volume

        Sum the volume of all `DynamicCompartment`s that are not extracellular.

        Returns:
            :obj:`float`: the cell's volume (L)
        """
        return sum([dynamic_compartment.volume() for dynamic_compartment in self.cellular_dyn_compartments])

    def cell_dry_weight(self):
        """ Compute the cell's dry weight

        Returns:
            :obj:`float`: the cell's dry weight (g)
        """
        if self.water_in_model:
            return self.fraction_dry_weight * self.cell_mass()
        else:
            return self.cell_mass()

    def get_growth(self):
        """ Report the cell's growth in cell/s, relative to the cell's initial volume

        Returns:
            (:obj:`float`): growth in cell/s, relative to the cell's initial volume
        """
        # TODO(Arthur): implement growth
        pass

    def get_aggregate_state(self):
        """ Report the cell's aggregate state

        Returns:
            :obj:`dict`: the cell's aggregate state
        """
        aggregate_state = {
            'cell mass': self.cell_mass(),
            'cell volume': self.cell_volume()
        }

        compartments = {}
        for dynamic_compartment in self.cellular_dyn_compartments:
            compartments[dynamic_compartment.id] = {
                'name': dynamic_compartment.name,
                'mass': dynamic_compartment.mass(),
                'volume': dynamic_compartment.volume(),
            }
        aggregate_state['compartments'] = compartments
        return aggregate_state

    def eval_dynamic_observables(self, time, observables_to_eval=None):
        """ Evaluate some dynamic observables at time `time`

        Args:
            time (:obj:`float`): the simulation time
            observables_to_eval (:obj:`list` of `str`, optional): if provided, ids of the observables to
                evaluate; otherwise, evaluate all observables

        Returns:
            :obj:`dict`: map from the IDs of dynamic observables in `observables_to_eval` to their
                values at simulation time `time`
        """
        if observables_to_eval is None:
            observables_to_eval = list(self.dynamic_observables.keys())
        evaluated_observables = {}
        for dyn_observable_id in observables_to_eval:
            evaluated_observables[dyn_observable_id] = self.dynamic_observables[dyn_observable_id].eval(time)
        return evaluated_observables

    def get_num_submodels(self):
        """ Provide the number of submodels

        Returns:
            :obj:`int`: the number of submodels
        """
        return self.num_submodels

    def set_stop_condition(self, simulation):
        """ Set the simulation's stop_condition

        A simulation's stop condition is constructed as a logical 'or' of all `StopConditions` in
        a model.

        Args:
            simulation (:obj:`SimulationEngine`): a simulation
        """
        if self.dynamic_stop_conditions:
            dynamic_stop_conditions = self.dynamic_stop_conditions.values()
            def stop_condition(time):
                for dynamic_stop_condition in dynamic_stop_conditions:
                    print('checking dynamic_stop_condition', dynamic_stop_condition)
                    if dynamic_stop_condition.eval(time):
                        return True
                return False
            simulation.set_stop_condition(stop_condition)

    def get_species_count_array(self, now):     # pragma no cover   not used
        """ Map current species counts into an np array

        Args:
            now (:obj:`float`): the current simulation time

        Returns:
            numpy array, #species x # compartments, containing count of specie in compartment
        """
        species_counts = np.zeros((len(model.species), len(model.compartments)))
        for species in model.species:
            for compartment in model.compartments:
                specie_id = Species.gen_id(species, compartment)
                species_counts[ species.index, compartment.index ] = \
                    model.local_species_population.read_one(now, specie_id)
        return species_counts
