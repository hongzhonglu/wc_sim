""" A deterministic version of SSA for testing DynamicSubmodel and multialgorithmic simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-15
:Copyright: 2019, Karr Lab
:License: MIT
"""

from de_sim.simulation_message import SimulationMessage
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel


class ExecuteDsaReaction(SimulationMessage):
    """ A simulation message sent by a :obj:`DsaSubmodel` instance to itself

    Provides data needed to execute a Deterministic Simulation Algorithm reaction.

    Attributes:
        reaction_index (:obj:`int`): index of the selected reaction in
            `DsaSubmodel.reactions`.
    """
    attributes = ['reaction_index']


class DsaSubmodel(DynamicSubmodel):
    """ Init a :obj:`DsaSubmodel`

    The Deterministic Simulation Algorithm (DSA) is a deterministic version of the Stochastic Simulation
    Algorithm. Each reaction executes deterministically at the rate determined by its rate law,
    which is achieved by scheduling the next execution of a reaction when the reaction executes.
    E.g., if reaction `R` executes at time `t` and `R`\ 's rate law has a rate of
    `r` then, the next execution of `R` will occur at time `t + 1/r`.

    Attributes:
        reaction_table (:obj:`dict`): map from reaction id to reaction index in `self.reactions`
    """

    # the message type sent by DsaSubmodel
    SENT_MESSAGE_TYPES = [ExecuteDsaReaction]

    # register the message types sent
    messages_sent = SENT_MESSAGE_TYPES

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ExecuteDsaReaction]

    event_handlers = [(ExecuteDsaReaction, 'handle_ExecuteDsaReaction_msg')]

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population, options=None):
        """ Initialize a DSA submodel instance

        Args:
            id (:obj:`str`): unique id of this dynamic DSA submodel
            dynamic_model (:obj: `DynamicModel`): the aggregate state of a simulation
            reactions (:obj:`list` of `wc_lang.Reaction`): the reactions modeled by this DSA submodel
            species (:obj:`list` of `wc_lang.Species`): the species that participate in the reactions
                modeled by this DSA submodel
            dynamic_compartments (:obj: `dict`): `DynamicCompartment`s, keyed by id, that contain
                species which participate in reactions that this DSA submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                DSA submodel's species population
            options (:obj:`dict`, optional): DSA submodel options
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        self.options = options
        self.reaction_table = {}
        for index, rxn in enumerate(self.reactions):
            self.reaction_table[rxn.id] = index

    def send_initial_events(self):
        """ Send this DSA submodel's initial events

        Schedule initial reactions at time `1 / (2 rate)` so that the linear approximation for the number
        of times a reaction with constant rate r has executed is `y = rate x`.

        This method overrides a :obj:`DynamicSubmodel` method.
        """
        # todo: don't schedule reactions that can't execute - requires predictions of the future populations
        for reaction in self.reactions:
            rate = self.calc_reaction_rate(reaction)
            dt = 1.0/(2 * rate)
            reaction_index = self.reaction_table[reaction.id]
            self.schedule_ExecuteDsaReaction(dt, reaction_index)

    def handle_ExecuteDsaReaction_msg(self, event):
        """ Handle a simulation event that contains an :obj:`ExecuteDsaReaction` message

        Args:
            event (:obj:`Event`): an :obj:`Event` to execute

        Raises:
            :obj:`DynamicMultialgorithmError:` if the reaction does not have sufficient reactants to execute
        """
        # reaction_index is the reaction to execute
        reaction_index = event.message.reaction_index
        # execute reaction if it is enabled
        reaction = self.reactions[reaction_index]
        if not self.enabled_reaction(reaction):
            raise DynamicMultialgorithmError(self.time, f"Insufficient reactants to execute reaction {reaction.id}")
        self.execute_reaction(reaction)
        self.schedule_next_reaction_execution(reaction)

    def schedule_ExecuteDsaReaction(self, dt, reaction_index):
        """ Schedule an :obj:`ExecuteDsaReaction` event.

        Args:
            dt (:obj:`float`): simulation delay until the event containing :obj:`ExecuteDsaReaction` executes
            reaction_index (:obj:`int`): index of the reaction to execute
        """
        self.send_event(dt, self, ExecuteDsaReaction(reaction_index))

    def schedule_next_reaction_execution(self, reaction):
        """ Schedule the next execution of a reaction

        Args:
            reaction (:obj:`Reaction`): the reaction being scheduled
        """
        # todo: don't schedule reactions that can't execute - requires predictions of the future populations
        rate = self.calc_reaction_rate(reaction)
        dt = 1.0/rate
        reaction_index = self.reaction_table[reaction.id]
        self.schedule_ExecuteDsaReaction(dt, reaction_index)
