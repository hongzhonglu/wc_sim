""" A generic dynamic submodel; a multi-algorithmic model is constructed from multiple dynamic submodel subclasses

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Author: Jonathan Karr, karr@mssm.edu
:Date: 2016-03-22
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import numpy as np
from scipy.constants import Avogadro

from wc_lang.core import Species, Reaction, Compartment, Parameter
from wc_lang.expression_utils import RateLawUtils
from wc_sim.multialgorithm.dynamic_components import DynamicCompartment, DynamicModel
from wc_sim.core.simulation_object import SimulationObject, ApplicationSimulationObject
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm import message_types, distributed_properties
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError, SpeciesPopulationError
from wc_sim.multialgorithm.dynamic_expressions import DynamicRateLaw, DynamicReaction


class DynamicSubmodel(ApplicationSimulationObject):
    """ Provide generic dynamic submodel functionality

    Subclasses of `DynamicSubmodel` are combined into a multi-algorithmic model.

    Attributes:
        id (:obj:`str`): unique id of this dynamic submodel / simulation object
        dynamic_model (:obj: `DynamicModel`): the aggregate state of a simulation
        reactions (:obj:`list` of `Reaction`): the reactions modeled by this dynamic submodel
        dynamic_reactions (:obj:`list` of `DynamicReaction`): the dynamic reactions modeled by this
            dynamic submodel: created during initializatino
        rates (:obj:`np.array`): array to hold reaction rates
        species (:obj:`list` of `Species`): the species that participate in the reactions modeled
            by this dynamic submodel
        parameters (:obj:`list` of `Parameter`): the model's parameters
        parameter_values (:obj:`dict` mapping `str` id to `float`): the values of all parameters
        dynamic_compartments (:obj:`dict` mapping `str` id to `DynamicCompartment`): the dynamic
            compartments containing species that participate in reactions that this dynamic submodel models,
            including adjacent compartments used by its transfer reactions
        local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
            dynamic submodel's species population
        logger (:obj:`logging.Logger`): debug logger
    """
    def __init__(self, id, dynamic_model, reactions, species, parameters, dynamic_compartments, local_species_population):
        """ Initialize a dynamic submodel
        """
        super().__init__(id)
        self.id = id
        self.dynamic_model = dynamic_model
        self.log_with_time("submodel: {}; reactions: {}".format(self.id,
            [reaction.id for reaction in reactions]))
        self.species = species
        self.parameters = parameters
        self.dynamic_compartments = dynamic_compartments
        self.local_species_population = local_species_population
        self.logger = debug_logs.get_log('wc.debug.file')
        self.reactions = reactions
        self.dynamic_reactions = []
        for rxn in self.reactions:
            self.dynamic_reactions.append(DynamicReaction(dynamic_model, local_species_population, rxn))
        self.initialize_optimizations()

    def initialize_optimizations(self):
        """ Initialize the data needed to optimize a submodel's calculations
        """
        # optimization: preallocate rates vector
        self.rates = np.full(len(self.dynamic_reactions), np.nan)
        # optimization: precompute parameter_values
        self.parameter_values = {param.id: param.value for param in self.parameters}
        # optimization: preallocate compartment volumes dictionary
        self.volumes = {}
        # optimization: precompute species ids
        self.species_ids = [s.id() for s in self.species]
        # optimization: precompute set of species ids
        self.set_species_ids = set(self.species_ids)

    # The next 2 methods implement the abstract methods in ApplicationSimulationObject
    def send_initial_events(self):
        pass    # pragma: no cover

    GET_STATE_METHOD_MESSAGE = 'object state to be provided by subclass'
    def get_state(self):
        return DynamicSubmodel.GET_STATE_METHOD_MESSAGE

    # At any time instant, event messages are processed in this order
    # TODO(Arthur): cover after MVP wc_sim done
    event_handlers = [(message_types.GetCurrentProperty, 'handle_get_current_prop_event')]  # pragma: no cover

    # TODO(Arthur): cover after MVP wc_sim done
    messages_sent = [message_types.GiveProperty]    # pragma: no cover

    def get_species_ids(self):
        """ Get ids of species used by this dynamic submodel

        Returns:
            :obj:`list`: ids of species used by this dynamic submodel
        """
        return self.species_ids

    def get_specie_counts(self):
        """ Get the current species counts for species used by this dynamic submodel

        Returns:
            :obj:`dict`: a map: species_id -> current copy number
        """
        return self.local_species_population.read(self.time, self.set_species_ids)

    def compute_volumes(self):
        """ Compute the volumes of the dynamic compartments that contain species used by reactions in this submodel

        Volumes are stored in `self.volumes`, a `dict` that maps compartment_id -> volume, for each compartment

        Returns:
            :obj:`None`:
        """
        for id, dynamic_compartment in self.dynamic_compartments.items():
            self.volumes[id] = dynamic_compartment.volume()

    '''
    harden and optimize:
    harden:
        if a compartment has counts > 0 but mass is 0 because some MWs == 0 -> raise exception
        more convenient:
            prohibit species with mw == 0, compartments with initial density that's undefined or 0
    optimize:
        pre-compute and cache volumes of all compartments occupied by species in this submodel
            (actually, just volumes of species used by rate laws)
        pre-compute mappings: species_id -> dynamic compartment;
        pre-allocate a dict species_id -> concentration
        make static simulator-wide mapping of species to array index, and use for all species references
    '''
    def get_specie_concentrations(self):
        """ Get the current concentrations of species used by this dynamic submodel

        Concentrations are obtained from species counts.
        concentration ~ count/volume
        Provide concentrations for only species stored in this dynamic submodel's compartments, whose
        volume is known.

        Returns:
            :obj:`dict`: a map: species_id -> species concentration

        Raises:
            :obj:`MultialgorithmError:` if a dynamic compartment cannot be found for a specie being modeled,
                or if the compartments volume is 0
        """
        counts = self.get_specie_counts()
        # if all counts are 0, concentrations are too
        if 0 == sum(counts.values()):
            return {specie_id:0.0 for specie_id in self.get_species_ids()}

        self.compute_volumes()
        concentrations = {}
        for specie_id in self.get_species_ids():
            (_, compartment_id) = get_species_and_compartment_from_name(specie_id)
            if compartment_id not in self.dynamic_compartments:
                raise MultialgorithmError("dynamic submodel '{}' lacks dynamic compartment '{}' for specie '{}'".format(
                    self.id, compartment_id, specie_id))
            dynamic_compartment = self.dynamic_compartments[compartment_id]
            # todo: optimize: convert to assert that's compiled out at run-time
            if self.volumes[compartment_id] == 0:
                raise MultialgorithmError("dynamic submodel '{}' cannot compute concentration in "
                    "compartment '{}' with volume=0".format(self.id, compartment_id))

            concentrations[specie_id] = counts[specie_id]/(self.volumes[compartment_id]*Avogadro)
        return concentrations

    def get_parameter_values(self):
        """ Get the current parameter values for this dynamic submodel

        Returns:
            :obj:`dict`: a map: parameter_id -> parameter value
        """
        return self.parameter_values

    def get_num_submodels(self):
        """ Provide the number of submodels

        Returns:
            :obj:`int`: the number of submodels
        """
        return self.dynamic_model.get_num_submodels()

    '''
    harden and optimize:
    harden:
        can we straightforwardly model a MM rate in a SSA submodel?
    optimize:
        pre-categorize reactions & rate laws into these types, and only calculate concentrations which are needed
            reactions without rate laws: needs nothing
            mass action: needs volume (except for reactions of order 1)
            MM and other: needs concentrations which needs volume
        pre-compute compartment volume(s) and pass to dynamic_rate_law.eval()
        compute rates only as necessary: just for rate laws whose dynamic expressions have changed
            statically, construct dependencies among dynamic expressions, with species counts as the leaves
                make static map from each species to the rate laws that depend on it
            at each rates calculation, form union of rate laws that depend on all changed species counts
            compute those rate laws
    '''
    def calc_reaction_rates(self):
        """ Calculate the rates for this dynamic submodel's reactions

        Rates computed by eval'ing reactions provided in this dynamic submodel's definition,
        with species concentrations obtained by lookup from the dict
        `species_concentrations`. This assumes that all reversible reactions have been split
        into two forward reactions, as is done by `wc_lang.transform.SplitReversibleReactionsTransform`.

        Returns:
            :obj:`np.ndarray`: a numpy array of reaction rates, indexed by reaction index
        """
        species_concentrations = self.get_specie_concentrations()
        for idx_reaction, dynamic_rxn in enumerate(self.dynamic_reactions):
            if hasattr(dynamic_rxn, 'dynamic_rate_law'):
                self.rates[idx_reaction] = dynamic_rxn.dynamic_rate_law.eval(self.time, self.get_parameter_values(),
                    species_concentrations)
        # TODO(Arthur): optimization: get this if to work:
        # if self.logger.isEnabledFor(self.logger.getEffectiveLevel()):
        # print('self.logger.getEffectiveLevel())', self.logger.getEffectiveLevel())
        # msg = str([(self.dynamic_reactions[i].id, self.rates[i]) for i in range(len(self.dynamic_reactions))])
        # debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time)
        return self.rates

    # These methods - enabled_reaction, identify_enabled_reactions, execute_reaction - are used
    # by discrete time submodels like SSASubmodel and the SkeletonSubmodel.
    def enabled_reaction(self, reaction):
        """ Determine whether the cell state has adequate specie counts to run a reaction

        Indicate whether the current specie counts are large enough to execute `reaction`, based on
        its stoichiometry.

        Args:
            reaction (:obj:`Reaction`): the reaction to evaluate

        Returns:
            :obj:`bool`: True if `reaction` is stoichiometrically enabled
        """
        for participant in reaction.participants:
            species_id = Species.gen_id(participant.species.species_type,
                participant.species.compartment)
            count = self.local_species_population.read_one(self.time, species_id)
            # 'participant.coefficient < 0' determines whether the participant is a reactant
            is_reactant = participant.coefficient < 0
            if is_reactant and count < -participant.coefficient:
                return False
        return True

    def identify_enabled_reactions(self):
        """ Determine which reactions have adequate specie counts to run

        Returns:
            np array: an array indexed by reaction number; 0 indicates reactions without adequate
                species counts
        """
        enabled = np.full(len(self.reactions), 1)
        for idx_reaction, rxn in enumerate(self.reactions):
            if not self.enabled_reaction(rxn):
                enabled[idx_reaction] = 0

        return enabled

    def execute_reaction(self, reaction):
        """ Update species counts to reflect the execution of a reaction

        Called by discrete submodels, like SSA. Counts are updated in the `AccessSpeciesPopulations`
        that store them.

        Args:
            reaction (:obj:`Reaction`): the reaction being executed

        Raises:
            :obj:`MultialgorithmError:` if the species population cannot be updated
        """
        adjustments = {}
        for participant in reaction.participants:
            species_id = Species.gen_id(participant.species.species_type,
                participant.species.compartment)
            if not species_id in adjustments:
                adjustments[species_id] = 0
            adjustments[species_id] += participant.coefficient
        try:
            self.local_species_population.adjust_discretely(self.time, adjustments)
        except SpeciesPopulationError as e:
            raise MultialgorithmError("{:7.1f}: dynamic submodel '{}' cannot execute reaction: {}: {}".format(
                self.time, self.id, reaction.id, e))

    # TODO(Arthur): cover after MVP wc_sim done
    def handle_get_current_prop_event(self, event):   # pragma: no cover
        """ Handle a GetCurrentProperty simulation event.

        Args:
            event (:obj:`wc_sim.core.Event`): an `Event` to process

        Raises:
            MultialgorithmError: if an `GetCurrentProperty` message requests an unknown property
        """
        property_name = event.message.property_name
        if property_name == distributed_properties.MASS:
            '''
            # TODO(Arthur): rethink this, as, strictly speaking, a dynamic submodel doesn't have mass, but its compartment does
                    self.send_event(0, event.sending_object, message_types.GiveProperty,
                        message=message_types.GiveProperty(property_name, self.time,
                            self.mass()))
            '''
            raise MultialgorithmError("Error: not handling distributed_properties.MASS")
        else:
            raise MultialgorithmError("Error: unknown property_name: '{}'".format(property_name))
