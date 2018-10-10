"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-10-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest
import os
import numpy as np

from wc_lang.core import (Model, Submodel,  SpeciesType, Species, Reaction, Compartment,
                          SpeciesCoefficient, Parameter, RateLaw, RateLawEquation,
                          SubmodelAlgorithm)
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.submodels.ssa import SSASubmodel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.message_types import GivePopulation, ExecuteSsaReaction
from wc_sim.multialgorithm.make_models import MakeModels, RateLawType
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError, FrozenSimulationError


class TestSsaSubmodel(unittest.TestCase):

    def setUp(self):
        self.model_1_species = MakeModels.make_test_model('1 species, 1 reaction')
        self.ssa_submodel = self.make_ssa_submodel(self.model_1_species)

        self.spec_type_0_cn = spec_type_0_cn = 1000000
        self.specie_copy_numbers={
            'spec_type_0[compt_1]':spec_type_0_cn,
            'spec_type_1[compt_1]':2*spec_type_0_cn
        }
        model = MakeModels.make_test_model('1 species, 1 reaction',
            specie_copy_numbers={'spec_type_0[compt_1]':1})
        self.ssa_submodel_zero_pop = self.make_ssa_submodel(model)
        self.ssa_submodel_zero_pop.local_species_population.adjust_discretely(0, {'spec_type_0[compt_1]': -1})

    def make_ssa_submodel(self, model, default_center_of_mass=None):
        multialgorithm_simulation = MultialgorithmSimulation(model, {})
        simulation, _ = multialgorithm_simulation.build_simulation()
        wc_lang_ssa_submodel = model.submodels[0]
        ssa_submodel = SSASubmodel(
            model.id,
            multialgorithm_simulation.dynamic_model,
            list(wc_lang_ssa_submodel.reactions),
            wc_lang_ssa_submodel.get_species(),
            wc_lang_ssa_submodel.parameters,
            multialgorithm_simulation.get_dynamic_compartments(wc_lang_ssa_submodel),
            multialgorithm_simulation.local_species_population,
            default_center_of_mass=default_center_of_mass)
        ssa_submodel.add(simulation)
        return ssa_submodel

    def test_SSA_submodel_init(self):
        ssa_submodel = self.make_ssa_submodel(self.model_1_species, default_center_of_mass=20)
        self.assertTrue(isinstance(ssa_submodel, SSASubmodel))

    def test_determine_reaction_propensities(self):
        # static tests of ssa methods
        self.assertEqual(self.ssa_submodel.num_SsaWaits, 0)
        self.assertTrue(0 < self.ssa_submodel.ema_of_inter_event_time.get_ema())
        propensities, total_propensities = self.ssa_submodel.determine_reaction_propensities()
        # there's only one reaction
        self.assertEqual(propensities[0], total_propensities)

        model = MakeModels.make_test_model(
            '2 species, a pair of symmetrical reactions with constant rates',
            specie_copy_numbers=self.specie_copy_numbers)
        # order 1 reactions with MA rates proportional to reactant copy numbers
        ssa_submodel = self.make_ssa_submodel(model)
        propensities, _ = ssa_submodel.determine_reaction_propensities()
        self.assertEqual(2 * propensities[0], propensities[1])

        with self.assertRaisesRegexp(FrozenSimulationError,
            '1 SSA submodel .* with total propensities = 0 cannot progress'):
            self.ssa_submodel_zero_pop.determine_reaction_propensities()

    def test_get_reaction_propensities(self):
        propensities, total_propensities = self.ssa_submodel.get_reaction_propensities()
        self.assertEqual(propensities[0], total_propensities)

        self.assertEqual((None, None), self.ssa_submodel_zero_pop.get_reaction_propensities())
        # next event should be at infinite time
        self.assertEqual(self.ssa_submodel_zero_pop.simulator.event_queue.next_event_time(), float('inf'))

        model_w_2_submodels = MakeModels.make_test_model('1 species, 1 reaction',
            specie_copy_numbers={'spec_type_0[compt_1]':1},
            num_submodels=2)
        submodel_0 = self.make_ssa_submodel(model_w_2_submodels)
        submodel_0.local_species_population.adjust_discretely(0, {'spec_type_0[compt_1]': -1})
        submodel_0.get_reaction_propensities()
        # next event is EMA in the future
        self.assertEqual(submodel_0.simulator.event_queue.next_event_time(),
            submodel_0.ema_of_inter_event_time.get_ema())

    def test_schedule_next_SSA_reaction(self):
        # no initial events are scheduled
        self.assertTrue(self.ssa_submodel.simulator.event_queue.empty())
        # will schedule one ExecuteSsaReaction
        dt = self.ssa_submodel.schedule_next_SSA_reaction()
        self.assertTrue(0 < dt)
        next_event = self.ssa_submodel.simulator.event_queue.next_events()[0]
        self.assertEqual(next_event.creation_time, 0)
        self.assertEqual(next_event.event_time, dt)
        self.assertEqual(next_event.sending_object, self.ssa_submodel)
        self.assertEqual(next_event.receiving_object, self.ssa_submodel)
        self.assertEqual(type(next_event.message), message_types.ExecuteSsaReaction)

        # no initial events are scheduled
        self.assertTrue(self.ssa_submodel_zero_pop.simulator.event_queue.empty())
        self.assertEqual(None, self.ssa_submodel_zero_pop.schedule_next_SSA_reaction())
        # single submodel with pop=0 schedules SsaWait at infinity
        self.assertEqual(self.ssa_submodel_zero_pop.simulator.event_queue.next_event_time(), float('inf'))

    def test_schedule_next_events(self):
        self.ssa_submodel.schedule_next_events()
        next_event = self.ssa_submodel.simulator.event_queue.next_events()[0]
        self.assertEqual(type(next_event.message), message_types.ExecuteSsaReaction)

    def test_handle_ExecuteSsaReaction_msg(self):
        self.ssa_submodel.schedule_next_events()
        next_event = self.ssa_submodel.simulator.event_queue.next_events()[0]
        lsp = self.ssa_submodel.local_species_population
        specie_id = 'spec_type_0[compt_1]'
        init_pop = lsp.read(0, set([specie_id]))[specie_id]
        # execute the ExecuteSsaReaction msg, which reduced pop by 1
        self.ssa_submodel.handle_ExecuteSsaReaction_msg(next_event)
        final_pop = lsp.read(0, set([specie_id]))[specie_id]
        self.assertEqual(init_pop-1, final_pop)

    # todo: complete testing ssa.py
    '''
    def test_handle_SsaWait_msg(self):
        self.ssa_submodel.schedule_next_events()
        next_event = self.ssa_submodel.simulator.event_queue.next_events()[0]
    '''

    def test_execute_SSA_reaction(self):
        # with rates given by reactant population, propensities proportional to copy number
        model = MakeModels.make_test_model(
            '2 species, a pair of symmetrical reactions rates given by reactant population',
            specie_copy_numbers=self.specie_copy_numbers)
        ssa_submodel = self.make_ssa_submodel(model)
        propensities, _ = ssa_submodel.determine_reaction_propensities()
        self.assertEqual(2*propensities[0], propensities[1])
        ssa_submodel.execute_SSA_reaction(0)
        population = ssa_submodel.local_species_population.read(0,
            set(['spec_type_0[compt_1]', 'spec_type_1[compt_1]']))
        expected_population = {
            'spec_type_0[compt_1]': self.spec_type_0_cn-1,
            'spec_type_1[compt_1]': 2 * self.spec_type_0_cn+1
        }
        self.assertEqual(population, expected_population)
