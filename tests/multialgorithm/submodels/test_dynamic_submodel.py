"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest
import os, sys
import six
from builtins import super
import warnings

from scipy.constants import Avogadro

import wc_lang
from wc_lang.io import Reader
from wc_lang.core import Reaction, SpeciesType, Species
from wc_lang.prepare import PrepareModel, CheckModel
from wc_lang.transform import SplitReversibleReactionsTransform
from wc_sim.core.simulation_object import SimulationObject, SimulationObjectInterface
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm import message_types, distributed_properties
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
from tests.core.mock_simulation_object import MockSimulationObjectInterface


class MockSimulationObject(MockSimulationObjectInterface):

    def send_initial_events(self): pass

    def handle_GiveProperty_event(self, event):
        """Perform a unit test on the molecular weight of a SpeciesPopSimObject"""
        property_name = event.event_body.property_name
        self.test_case.assertEqual(property_name, distributed_properties.MASS)
        self.test_case.assertEqual(event.event_body.value, self.expected_value)

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class, [
            (message_types.GiveProperty, this_class.handle_GiveProperty_event)])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, [message_types.GetCurrentProperty])


class TestDynamicSubmodel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_submodel_no_shared_species.xlsx')

    def setUp(self):
        warnings.simplefilter("ignore")
        SpeciesType.objects.reset()

        self.model = Reader().run(self.MODEL_FILENAME)
        # TODO:(Arthur): put these in a high-level prepare
        SplitReversibleReactionsTransform().run(self.model)
        PrepareModel(self.model).run()
        CheckModel(self.model).run()

        self.local_species_pop = local_species_pop = MultialgorithmSimulation.create_local_species_population(self.model)
        self.dynamic_submodels = {}
        for lang_submodel in self.model.get_submodels():
            dynamic_compartments = MultialgorithmSimulation.create_dynamic_compartments_for_submodel(
                self.model,
                lang_submodel,
                local_species_pop)
            self.dynamic_submodels[lang_submodel.id] = DynamicSubmodel(
                lang_submodel.id,
                lang_submodel.reactions,
                lang_submodel.get_species(),
                self.model.get_parameters(),
                dynamic_compartments,
                local_species_pop)

    def test_get_specie_concentrations(self):
        expected_conc = {}
        for conc in self.model.get_concentrations():
            expected_conc[
                Species.gen_id(conc.species.species_type, conc.species.compartment)] = conc.value
        for dynamic_submodel in self.dynamic_submodels.values():
            for specie_id,v in dynamic_submodel.get_specie_concentrations().items():
                if specie_id in expected_conc:
                    self.assertAlmostEqual(expected_conc[specie_id], v)

    def test_calc_reaction_rates(self):
        expected_rates = {'reaction_2': 0.0, 'reaction_4': 2.0}
        for dynamic_submodel in self.dynamic_submodels.values():
            rates = dynamic_submodel.calc_reaction_rates()
            for index,rxn in enumerate(dynamic_submodel.reactions):
                if rxn.id in expected_rates:
                    self.assertAlmostEqual(list(rates)[index], expected_rates[rxn.id])

    def test_enabled_reaction(self):
        expected_enabled = {
            'submodel_1': set(['reaction_1', '__exchange_reaction__1', '__exchange_reaction__2']),
            'submodel_2': set(['reaction_4', 'reaction_3_forward', 'reaction_3_backward']),
        }
        for dynamic_submodel in self.dynamic_submodels.values():
            enabled = set()
            for rxn in dynamic_submodel.reactions:
                if dynamic_submodel.enabled_reaction(rxn):
                    enabled.add(rxn.id)
            self.assertEqual(expected_enabled[dynamic_submodel.id], enabled)

    # TODO(Arthur): next;
    def test_identify_enabled_reactions(self):
        pass

    # TODO(Arthur): next;
    def test_execute_reaction(self):
        # test one reaction 'by hand'
        adjustments = {'reaction_3': {'specie_2[c]':-1, 'specie_4[c]':-2, 'specie_5[c]':1}}
        for dynamic_submodel in self.dynamic_submodels.values():
            for reaction in dynamic_submodel.reactions:
                print(reaction.id, dynamic_submodel.reactions)
                if reaction.id in adjustments:
                    before = dynamic_submodel.get_specie_counts()
                    dynamic_submodel.execute_reaction(reaction)
                    after = dynamic_submodel.get_specie_counts()
                    for specie_id in adjustments[reaction.id].keys():
                        self.assertEqual(adjustments[reaction.id][specie_id],
                            after[specie_id]-before[specie_id])


# TODO(Arthur): test submodels with running simulator
# TODO(Arthur): eliminate redundant code in test_submodel.py; search for distributed_properties
"""
class TestDynamicSubmodelSimulating(unittest.TestCase):

    def test_dynamic_submodel(self):
        pass
"""