""" Test species_populations.py

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-04
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import numpy as np
from numpy import all
import os, unittest, copy
import re
import string
import sys
import unittest

from scipy.constants import Avogadro
from scipy.stats import binom

from wc_lang.io import Reader
import wc_lang
from wc_sim.core.errors import SimulatorError
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.core.simulation_message import SimulationMessage
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.species_populations import (LOCAL_POP_STORE, DynamicSpecie, SpeciesPopSimObject,
    SpeciesPopulationCache, LocalSpeciesPopulation, MakeTestLSP, AccessSpeciesPopulations)
from wc_sim.multialgorithm.multialgorithm_errors import NegativePopulationError, SpeciesPopulationError
from wc_sim.multialgorithm import distributed_properties
from wc_utils.util.rand import RandomStateManager
from tests.core.mock_simulation_object import MockSimulationObject

def store_i(i):
    return "store_{}".format(i)

def specie_l(l):
    return "specie_{}".format(l)

remote_pop_stores = {store_i(i):None for i in range(1, 4)}
species_ids = [specie_l(l) for l in list(string.ascii_lowercase)[0:5]]


class TestAccessSpeciesPopulations(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_model_for_access_species_populations.xlsx')

    # MODEL_FILENAME_STEADY_STATE contains a model with no net population change every 2 sec.
    MODEL_FILENAME_STEADY_STATE = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_model_for_access_species_populations_steady_state.xlsx')

    def setUp(self):
        self.an_ASP = AccessSpeciesPopulations(None, remote_pop_stores)
        self.simulator = SimulationEngine()

    def test_add_species_locations(self):

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        map = dict.fromkeys(species_ids[:2], store_i(1))
        self.assertEqual(self.an_ASP.species_locations, map)

        self.an_ASP.add_species_locations(store_i(2), species_ids[2:])
        map.update(dict(zip(species_ids[2:], [store_i(2)]*3)))
        self.assertEqual(self.an_ASP.species_locations, map)

        locs = self.an_ASP.locate_species(species_ids[1:4])
        self.assertEqual(locs[store_i(1)], {'specie_b'})
        self.assertEqual(locs[store_i(2)], {'specie_c', 'specie_d'})

        self.an_ASP.del_species_locations([specie_l('b')])
        del map[specie_l('b')]
        self.assertEqual(self.an_ASP.species_locations, map)
        self.an_ASP.del_species_locations(species_ids, force=True)
        self.assertEqual(self.an_ASP.species_locations, {})

    def test_add_species_locations(self):
        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.add_species_locations('no_such_store', species_ids[:2])
        self.assertIn("'no_such_store' not a known population store", str(cm.exception))

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        self.assertIn("species ['specie_a', 'specie_b'] already have assigned locations.",
            str(cm.exception))

        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.del_species_locations([specie_l('d'), specie_l('c')])
        self.assertIn("species ['specie_c', 'specie_d'] are not in the location map",
            str(cm.exception))

        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.locate_species([specie_l('d'), specie_l('c')])
        self.assertIn("species ['specie_c', 'specie_d'] are not in the location map",
            str(cm.exception))

    def test_other_exceptions(self):
        with self.assertRaises(SpeciesPopulationError) as cm:
            AccessSpeciesPopulations(None, {'a':None, LOCAL_POP_STORE:None})
        self.assertIn("{} not a valid remote_pop_store name".format(LOCAL_POP_STORE),
            str(cm.exception))
        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.read_one(0, 'no_such_specie')
        self.assertEqual(str(cm.exception), "read_one: specie 'no_such_specie' not in the location map.")

    @unittest.skip("skip until MultialgorithmSimulation().initialize() is ready")
    def test_population_changes(self):
        """ Test population changes that occur without using event messages."""
        self.set_up_simulation(self.MODEL_FILENAME)
        theASP = self.submodels['dfba_submodel'].access_species_population
        init_val=100
        self.assertEqual(theASP.read_one(0, 'specie_1[c]'), init_val)
        self.assertEqual(theASP.read(0, set(['specie_1[c]'])), {'specie_1[c]': init_val})

        with self.assertRaises(SpeciesPopulationError) as cm:
            theASP.read(0, set(['specie_2[c]']))
        self.assertIn("read: species ['specie_2[c]'] not in cache.", str(cm.exception))

        adjustment=-10
        self.assertEqual(theASP.adjust_discretely(0, {'specie_1[c]':adjustment}),
            ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(0, 'specie_1[c]'), init_val+adjustment)

        with self.assertRaises(SpeciesPopulationError) as cm:
            theASP.read_one(0, 'specie_none')
        self.assertIn("read_one: specie 'specie_none' not in the location map.", str(cm.exception))

        self.assertEqual(sorted(theASP.adjust_discretely(0,
            {'specie_1[c]': adjustment, 'specie_2[c]': adjustment})),
                sorted(['shared_store_1', 'LOCAL_POP_STORE']))
        self.assertEqual(theASP.read_one(0, 'specie_1[c]'), init_val + 2*adjustment)

        self.assertEqual(theASP.adjust_continuously(1, {'specie_1[c]':(adjustment, 0)}),
            ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(1, 'specie_1[c]'), init_val + 3*adjustment)

        flux=1
        time=2
        delay=3
        self.assertEqual(theASP.adjust_continuously(time, {'specie_1[c]':(adjustment, flux)}),
            ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(time+delay, 'specie_1[c]'),
            init_val + 4*adjustment + delay*flux)

        with self.assertRaises(SpeciesPopulationError) as cm:
            theASP.prefetch(0, ['specie_1[c]', 'specie_2[c]'])
        self.assertIn("prefetch: 0 provided, but delay must be non-negative", str(cm.exception))

        self.assertEqual(theASP.prefetch(1, ['specie_1[c]', 'specie_2[c]']), ['shared_store_1'])

    """
    todo: replace this code with calls to MultialgorithmSimulation().initialize()
    def initialize_simulation(self, model_file):
        self.set_up_simulation(model_file)
        delay_to_first_event = 1.0/len(self.submodels)
        for name,submodel in self.submodels.items():

            # prefetch into caches
            submodel.access_species_population.prefetch(delay_to_first_event,
                submodel.get_species_ids())

            # send initial event messages
            msg_body = message_types.ExecuteSsaReaction(0)
            submodel.send_event(delay_to_first_event, submodel, message_types.ExecuteSsaReaction,
                msg_body)

            delay_to_first_event += 1/len(self.submodels)
    """

    def verify_simulation(self, expected_final_pops, sim_end):
        """ Verify the final simulation populations."""
        for specie_id in self.shared_species:
            pop = self.shared_pop_sim_obj['shared_store_1'].read_one(sim_end, specie_id)
            self.assertEqual(expected_final_pops[specie_id], pop)

        for submodel in self.submodels.values():
            for specie_id in self.private_species[submodel.name]:
                pop = submodel.access_species_population.read_one(sim_end, specie_id)
                self.assertEqual(expected_final_pops[specie_id], pop)

    @unittest.skip("skip until MultialgorithmSimulation().initialize() is ready")
    def test_simulation(self):
        """ Test a short simulation."""

        self.initialize_simulation(self.MODEL_FILENAME)

        # run the simulation
        sim_end=3
        self.simulator.simulate(sim_end)

        # test final populations
        # Expected changes, based on the reactions executed
        expected_changes="""
        specie	c	e
        specie_1	-2	0
        specie_2	-2	0
        specie_3	3	-2
        specie_4	0	-1
        specie_5	0	1"""

        expected_final_pops = copy.deepcopy(self.init_populations)
        for row in expected_changes.split('\n')[2:]:
            (specie, c, e) = row.strip().split()
            for com in 'c e'.split():
                id = wc_lang.core.Species.gen_id(specie, com)
                expected_final_pops[id] += float(eval(com))

        self.verify_simulation(expected_final_pops, sim_end)

    @unittest.skip("skip until MODEL_FILENAME_STEADY_STATE is migrated")
    def test_stable_simulation(self):
        """ Test a steady state simulation.

        MODEL_FILENAME_STEADY_STATE contains a model with no net population change every 2 sec.
        """
        self.initialize_simulation(self.MODEL_FILENAME_STEADY_STATE)

        # run the simulation
        sim_end=100
        self.simulator.simulate(sim_end)
        expected_final_pops = self.init_populations
        self.verify_simulation(expected_final_pops, sim_end)


class TestLocalSpeciesPopulation(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize(seed=123)

        species_nums = range(1, 5)
        self.species_type_ids = species_type_ids = list(map(lambda x: "specie_{}".format(x), species_nums))
        self.compartment_ids = compartment_ids = ['c1', 'c2']
        self.species_ids = species_ids = []
        for species_type_id in species_type_ids[:2]:
            for compartment_id in compartment_ids[:2]:
                species_ids.append(wc_lang.core.Species.gen_id(species_type_id, compartment_id))
        self.init_populations = dict(zip(species_ids, species_nums))
        self.molecular_weights = dict(zip(species_ids, species_nums))

        self.local_species_pop = LocalSpeciesPopulation('test', self.init_populations,
            self.molecular_weights, model_continuously=True)
        self.population_slope = 1
        self.local_species_pop.adjust_continuously(0, {id:self.population_slope for id in species_ids})

        self.local_species_pop_no_init_pop_slope = LocalSpeciesPopulation(
            'test', self.init_populations, self.molecular_weights, model_continuously=False)

    def test_init(self):
        self.assertEqual(self.local_species_pop_no_init_pop_slope._all_species(), set(self.species_ids))
        an_LSP = LocalSpeciesPopulation('test', {}, {}, record_history=False)
        an_LSP.init_cell_state_specie('s1', 2, model_continuously=False)
        self.assertEqual(an_LSP.read(0, {'s1'}), {'s1': 2})

        with self.assertRaisesRegexp(SpeciesPopulationError,
            "specie_id 's1' already stored by this LocalSpeciesPopulation"):
            an_LSP.init_cell_state_specie('s1', 2, model_continuously=False)

        with self.assertRaisesRegexp(SpeciesPopulationError, "history not recorded"):
            an_LSP.report_history()

        with self.assertRaisesRegexp(SpeciesPopulationError, "history not recorded"):
            an_LSP.history_debug()

        with self.assertRaisesRegexp(SpeciesPopulationError,
            "Cannot init LocalSpeciesPopulation because some species are missing weights"):
            LocalSpeciesPopulation('test', {'s1': 2, 's2': 1}, {})

    def test_optional_species_argument(self):
        self.assertEqual(self.local_species_pop_no_init_pop_slope.read(0), self.init_populations)
        self.assertEqual(self.local_species_pop_no_init_pop_slope.read(2), self.init_populations)
        self.assertEqual(self.local_species_pop_no_init_pop_slope._check_species(0, species=None), None)
        t = 3
        self.local_species_pop_no_init_pop_slope._update_access_times(t, species=None)
        for specie_id in self.local_species_pop_no_init_pop_slope._all_species():
            self.assertEqual(self.local_species_pop_no_init_pop_slope.last_access_time[specie_id], t)

    def test_read_one(self):
        test_specie = 'specie_2[c2]'
        self.assertEqual(self.local_species_pop_no_init_pop_slope.read_one(1, test_specie),
            self.init_populations[test_specie])
        with self.assertRaisesRegexp(SpeciesPopulationError,
            re.escape("request for population of unknown specie(s): 'unknown_specie_id'")):
            self.local_species_pop_no_init_pop_slope.read_one(2, 'unknown_specie_id')
        with self.assertRaisesRegexp(SpeciesPopulationError, "is an earlier access of specie\(s\)"):
            self.local_species_pop_no_init_pop_slope.read_one(0, test_specie)

    def reusable_assertions(self, the_local_species_pop, population_slope):
        # test both discrete and hybrid species

        with self.assertRaisesRegexp(SpeciesPopulationError, "must be a set"):
            the_local_species_pop._check_species(0, 2)

        with self.assertRaisesRegexp(SpeciesPopulationError,
            "request for population of unknown specie\(s\):"):
            the_local_species_pop._check_species(0, {'x'})

        # populations have not changed
        self.assertEqual(the_local_species_pop.read(0, set(self.species_ids)), self.init_populations)
        first_specie = self.species_ids[0]
        the_local_species_pop.adjust_discretely(0, {first_specie: 3})
        self.assertEqual(the_local_species_pop.read(0, {first_specie}), {first_specie: 4})

        if population_slope:
            # counts: 1 initialization + 3 discrete adjustment + 2*population_slope:
            self.assertEqual(the_local_species_pop.read(2, {first_specie}), {first_specie: 4+2*population_slope})
            the_local_species_pop.adjust_continuously(2, {first_specie:0})
            # counts: 1 initialization + 3 discrete adjustment + 2 population_slope = 6:
            self.assertEqual(the_local_species_pop.read(2, {first_specie}), {first_specie: 6})
            for species_id in self.species_ids:
                self.assertIn(species_id, str(the_local_species_pop))

    def test_discrete_and_hybrid(self):

        for (local_species_pop, population_slope) in [(self.local_species_pop, self.population_slope),
            (self.local_species_pop_no_init_pop_slope, None)]:
            self.reusable_assertions(local_species_pop, population_slope)

    def test_adjustment_exceptions(self):
        time = 1.0
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "adjust_discretely error\(s\) at time .*:\ndiscrete_adjustment: "
            "negative population predicted for '.*', with decline from .*"):
            self.local_species_pop.adjust_discretely(time, {id:-10 for id in self.species_ids})

        self.local_species_pop.adjust_continuously(time, {id:-10 for id in self.species_ids})
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "adjust_continuously error\(s\) at time .*:\n"
            "continuous_adjustment: .* negative population predicted .*"):
            self.local_species_pop.adjust_continuously(time + 1, {id:0 for id in self.species_ids})

    def test_mass(self):
        self.assertEqual(self.local_species_pop.compartmental_mass('no_such_compartment'), 0)
        # 'specie_1[c1]' is not in compartment 'c2'
        self.assertEqual(self.local_species_pop.compartmental_mass('c2', species_ids=['specie_1[c1]']), 0)

        total_mass_c1 = 0
        for species_id in self.species_ids:
            if '[c1]' in species_id:
                total_mass_c1 += self.init_populations[species_id] * self.molecular_weights[species_id]
        total_mass_c1 = total_mass_c1 / Avogadro
        self.assertAlmostEqual(self.local_species_pop.compartmental_mass('c1'), total_mass_c1, places=30)

        mass_of_specie_1_in_c1 = \
            self.init_populations['specie_1[c1]'] * self.molecular_weights['specie_1[c1]'] / Avogadro
        self.assertAlmostEqual(self.local_species_pop.compartmental_mass('c1', species_ids=['specie_1[c1]']),
            mass_of_specie_1_in_c1, places=30)

        unknown_species = 'specie_x[c1]'
        with self.assertRaisesRegexp(SpeciesPopulationError,
            re.escape("molecular weight not available for '{}'".format(unknown_species))):
            self.local_species_pop.compartmental_mass('c1', species_ids=[unknown_species])

    def test_invalid_weights(self):
        bad_molecular_weights = ['x', float('nan'), -2, 0]
        good_molecular_weights = [3, 1E5, 1E-5, 2.34]
        num_mws = len(bad_molecular_weights)+len(good_molecular_weights)
        species_ids = [str(i) for i in range(num_mws)]
        molecular_weights = dict(zip(species_ids, bad_molecular_weights+good_molecular_weights))
        init_populations = dict(zip(species_ids, [1]*num_mws))
        local_species_pop = LocalSpeciesPopulation('test_invalid_weights', init_populations, molecular_weights)
        ids_w_bad_mws = species_ids[:len(bad_molecular_weights)]
        self.assertEqual(local_species_pop.invalid_weights(), set(ids_w_bad_mws))
        ids_w_bad_or_no_mw = ['x'] + ids_w_bad_mws
        self.assertEqual(local_species_pop.invalid_weights(species_ids=ids_w_bad_or_no_mw), set(ids_w_bad_or_no_mw))

    def test_make_test_lsp(self):
        make_test_lsp = MakeTestLSP()
        self.assertEqual(make_test_lsp.num_species, MakeTestLSP.DEFAULT_NUM_SPECIES)
        self.assertEqual(make_test_lsp.all_pops, MakeTestLSP.DEFAULT_ALL_POPS)
        self.assertEqual(make_test_lsp.all_mol_weights, MakeTestLSP.DEFAULT_ALL_MOL_WEIGHTS)
        kwargs = dict(
            num_species=7,
            all_pops=3E4,
            all_mol_weights=1000
        )
        make_test_lsp = MakeTestLSP(**kwargs)
        self.assertEqual(make_test_lsp.num_species, kwargs['num_species'])
        self.assertEqual(make_test_lsp.all_pops, kwargs['all_pops'])
        self.assertEqual(make_test_lsp.all_mol_weights, kwargs['all_mol_weights'])
        self.assertEqual(make_test_lsp.local_species_pop.read_one(0, 'specie_1[comp_id]'), kwargs['all_pops'])
        name = 'foo'
        make_test_lsp_3 = MakeTestLSP(name=name, initial_population=make_test_lsp.initial_population)
        self.assertEqual(make_test_lsp_3.initial_population, make_test_lsp.initial_population)
        make_test_lsp_4 = MakeTestLSP(initial_population=make_test_lsp.initial_population,
            molecular_weights=make_test_lsp.molecular_weights)
        self.assertEqual(make_test_lsp_4.initial_population, make_test_lsp.initial_population)
        self.assertEqual(make_test_lsp_4.molecular_weights, make_test_lsp.molecular_weights)

    """
    todo: test the distributed property MASS
    def test_mass(self):
        self.mass = sum([self.initial_population[specie_id]*self.molecular_weight[specie_id]/Avogadro
            for specie_id in self.species_ids])
        mock_obj = MockSimulationObject('mock_name', self, None, self.mass)
        self.simulator.add_object(mock_obj)
        mock_obj.send_event(1, self.test_species_pop_sim_obj, message_types.GetCurrentProperty,
            message_types.GetCurrentProperty(distributed_properties.MASS))
        self.simulator.initialize()
        self.simulator.simulate(2)
    """


class TestSpeciesPopulationCache(unittest.TestCase):

    def setUp(self):
        kwargs = dict(num_species=4, all_pops=0, all_mol_weights=0)
        make_test_lsp = MakeTestLSP(**kwargs)
        self.species_ids = make_test_lsp.species_ids
        local_species_population = make_test_lsp.local_species_pop

        remote_pop_stores = {store_i(i):None for i in range(1, 4)}
        self.an_ASP = AccessSpeciesPopulations(local_species_population, remote_pop_stores)
        self.an_ASP.add_species_locations(store_i(1), self.species_ids)
        self.an_ASP.add_species_locations(LOCAL_POP_STORE, ["specie_0"])
        self.species_population_cache = self.an_ASP.species_population_cache

    def test_species_population_cache(self):
        populations = [x*10 for x in range(len(self.species_ids))]
        population_dict = dict(zip(self.species_ids, populations))
        self.species_population_cache.cache_population(1, population_dict)
        s = self.species_ids[0]
        self.assertEqual(self.species_population_cache.read_one(1, s), population_dict[s])
        self.assertEqual(self.species_population_cache.read(1, self.species_ids),
            population_dict)

    def test_species_population_cache_exceptions(self):
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.cache_population(1, {"specie_0": 3})
        self.assertIn("some species are stored in the AccessSpeciesPopulations's local store: "
            "['specie_0'].", str(context.exception))

        self.species_population_cache.cache_population(0, {"specie_1[comp_id]": 3})
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.cache_population(-1, {"specie_1[comp_id]": 3})
        self.assertIn("cache_population: caching an earlier population: specie_id: specie_1[comp_id]; "
            "current time: -1 <= previous time 0.", str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read_one(1, 'specie_none')
        self.assertIn("SpeciesPopulationCache.read_one: specie 'specie_none' not in cache.",
            str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read_one(1, 'specie_1[comp_id]')
        self.assertIn("cache age of 1 too big for read at time 1 of specie 'specie_1[comp_id]'",
            str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read(0, ['specie_none'])
        self.assertIn("SpeciesPopulationCache.read: species ['specie_none'] not in cache.",
            str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read(1, ['specie_1[comp_id]'])
        self.assertIn(".read: species ['specie_1[comp_id]'] not reading recently cached value(s)",
            str(context.exception))


class TestDynamicSpecie(unittest.TestCase):

    def setUp(self):
        self.random_state = RandomStateManager.instance()

    def test_specie(self):

        # DynamicSpecies modeled only by discrete submodel(s)
        pop = 10
        s1 = DynamicSpecie('specie', self.random_state, pop)
        self.assertEqual(s1.get_population(0), pop)
        pop += 1
        self.assertEqual(s1.discrete_adjustment(0, 1), pop)
        self.assertEqual(s1.get_population(1), pop)
        pop -= 1
        self.assertEqual(s1.discrete_adjustment(1, -1), pop)
        self.assertEqual(s1.get_population(2), pop)

        # DynamicSpecies modeled by both continuous and discrete
        pop = 2
        s2 = DynamicSpecie('specie_3', self.random_state, pop, modeled_continuously=True)
        pop += 3
        self.assertEqual(s2.discrete_adjustment(4, 3), pop)

        # ensure that round=False can return non-integer population
        s2.continuous_adjustment(6, 0.5)
        self.assertIn(s2.get_population(7), {pop, pop+1})
        pop += 0.5
        self.assertEqual(s2.get_population(7, round=False), pop)

        pop = 10
        s3 = DynamicSpecie('specie2', self.random_state, pop)
        self.assertEqual("specie_name: specie2; last_population: {}".format(pop), str(s3))
        self.assertRegex(s3.row(), 'specie2\t{}.*'.format(pop))

        s4 = DynamicSpecie('specie', self.random_state, pop, modeled_continuously=True)
        self.assertEqual("specie_name: specie; last_population: {}; continuous_time: None; "
            "population_slope: None".format(pop), str(s4))
        self.assertRegex(s4.row(), '^specie\t{}\..*$'.format(pop))
        time = 3
        slope = 2
        s4.continuous_adjustment(time, slope)
        self.assertRegex(s4.row(), '^specie\t{}\..*\t{}\..*\t{}\..*$'.format(pop, time, slope))

        now = 4
        pop += slope * (now - time)
        self.assertEqual(s4.continuous_adjustment(now, 1), pop)

        with self.assertRaisesRegexp(SpeciesPopulationError,
            re.escape("continuous_adjustment(): adjustment_time is earlier than latest prior adjustment")):
            s4.continuous_adjustment(0, 0)

        # ensure that continuous_adjustment() returns an integral population
        s4.continuous_adjustment(6, 0.5)
        adjusted_pop = s4.get_population(7)
        self.assertEqual(int(adjusted_pop), adjusted_pop)

        self.assertRegex(DynamicSpecie.heading(), 'specie_name\t.*')

        # raise asserts
        with self.assertRaisesRegexp(AssertionError, 'DynamicSpecie .*: population should be >= 0'):
            DynamicSpecie('specie', self.random_state, -10)
        with self.assertRaisesRegexp(AssertionError,
            "DynamicSpecie '.*': initial discretely modeled population must be a non-negative integer, but .* isn't"):
            DynamicSpecie('specie', self.random_state, 1.5)
        ds = DynamicSpecie('specie', self.random_state, 1)
        with self.assertRaisesRegexp(AssertionError,
            "DynamicSpecie '.*': population_change must be an integer, but .* isn't"):
            ds.discrete_adjustment(2, .5)

        s5 = DynamicSpecie('s5', self.random_state, 10)
        with self.assertRaisesRegexp(SpeciesPopulationError,
            re.escape("continuous_adjustment(): DynamicSpecie not modeled by a continuous submodel")):
            s5.continuous_adjustment(0, 0)

    def test_first_continuous_adjustment(self):

        # DynamicSpecies modeled by both continuous and discrete
        pop = 1
        ds_hybrid = DynamicSpecie('ds_hybrid', self.random_state, pop, modeled_continuously=True)
        self.assertTrue(ds_hybrid.continuous_time is None)
        time = 0
        self.assertEqual(ds_hybrid.get_population(time), pop)
        time = 1
        # first continuous adjustment
        ds_hybrid.continuous_adjustment(time, 0.5)
        self.assertTrue(ds_hybrid.continuous_time is not None)
        self.assertEqual(ds_hybrid.continuous_time, time)
        time = 2
        pops = set()
        # get_population() does interpolation and rounding
        for _ in range(100):
            pops.add(ds_hybrid.get_population(time))
        self.assertEqual(pops, {1, 2})
        # another continuous adjustment
        time = 3
        ds_hybrid.continuous_adjustment(time, 0)
        # adds 1 molecule since the last continuous adjustment
        self.assertEqual(ds_hybrid.get_population(time), pop + 1)

    def test_validation(self):

        ### DynamicSpecies modeled only by a discrete submodel ###
        ds_discrete = DynamicSpecie('ds_discrete', self.random_state, 0)
        self.assertTrue(ds_discrete.last_adjustment_time == ds_discrete.last_read_time == -float('inf'))
        time = 1
        ds_discrete.get_population(time)
        self.assertEqual(ds_discrete.last_read_time, time)
        time = 3
        ds_discrete.discrete_adjustment(time, 0)
        self.assertEqual(ds_discrete.last_adjustment_time, time)

        # make exceptions
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "get_population\(\): read_time is earlier than latest prior adjustment: "):
            ds_discrete.get_population(time-1)

        time = 5
        ds_discrete.get_population(time)
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "discrete_adjustment\(\): adjustment_time is earlier than latest prior read: "):
            ds_discrete.discrete_adjustment(time-1, 0)

        time = 6
        ds_discrete.discrete_adjustment(time, 0)
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "discrete_adjustment\(\): adjustment_time is earlier than latest prior adjustment: "):
            ds_discrete.discrete_adjustment(time-1, 0)

        ### DynamicSpecies modeled by both continuous and discrete ###
        ds_hybrid = DynamicSpecie('ds_hybrid', self.random_state, 0, modeled_continuously=True)
        time = 0
        ds_hybrid.continuous_adjustment(time, 0)
        self.assertEqual(ds_hybrid.last_adjustment_time, time)

        # make exceptions
        time = 2
        ds_hybrid.get_population(time)
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "continuous_adjustment\(\): adjustment_time is earlier than latest prior read: "):
            ds_hybrid.continuous_adjustment(time-1, 0)

        time = 4
        ds_hybrid.continuous_adjustment(time, 0)
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "continuous_adjustment\(\): adjustment_time is earlier than latest prior adjustment: "):
            ds_hybrid.continuous_adjustment(time-1, 0)

        time = 6
        ds_hybrid.discrete_adjustment(time, 0)
        with self.assertRaisesRegexp(SpeciesPopulationError,
            "continuous_adjustment\(\): adjustment_time is earlier than latest prior adjustment: "):
            ds_hybrid.continuous_adjustment(time-1, 0)

    def test_species_with_and_wo_interpolation(self):
        # test interpolate control
        pop = 10
        s0 = DynamicSpecie('specie', self.random_state, pop, modeled_continuously=True)
        s0.continuous_adjustment(0, 1)
        self.assertEqual(s0.get_population(time=1), pop + 1)
        self.assertEqual(s0.get_population(time=1, interpolate=False), pop)
        self.assertEqual(s0.get_population(time=1, interpolate=True), pop + 1)

        # set the config interpolate variable to False
        from wc_sim.multialgorithm.species_populations import config_multialgorithm
        existing_interpolate = config_multialgorithm['interpolate']
        config_multialgorithm['interpolate'] = False

        s1 = DynamicSpecie('specie', self.random_state, pop, modeled_continuously=True)
        self.assertEqual(s1.get_population(time=1), pop)
        self.assertEqual(s0.get_population(time=1, interpolate=False), pop)
        self.assertEqual(s0.get_population(time=1, interpolate=True), pop + 1)

        # change the config interpolate variable back because all imports may already have been cached
        config_multialgorithm['interpolate'] = existing_interpolate

    def test_NegativePopulationError(self):
        s='specie_3'
        args = ('m', s, 2, -4.0)
        n1 = NegativePopulationError(*args)
        self.assertEqual(n1.specie, s)
        self.assertEqual(n1, NegativePopulationError(*args))
        n1.last_population += 1
        self.assertNotEqual(n1, NegativePopulationError(*args))
        self.assertTrue(n1.__ne__(NegativePopulationError(*args)))
        self.assertFalse(n1 == 3)

        p = "m: negative population predicted for 'specie_3', with decline from 3 to -1"
        self.assertEqual(str(n1), p)
        n1.delta_time=2
        self.assertEqual(str(n1), p + " over 2 time units")
        n1.delta_time=1
        self.assertEqual(str(n1), p + " over 1 time unit")

        d = {n1:1}
        self.assertTrue(n1 in d)

    def test_raise_NegativePopulationError(self):
        s1 = DynamicSpecie('specie_3', self.random_state, 2)

        time = 0
        with self.assertRaises(NegativePopulationError) as context:
            s1.discrete_adjustment(time, -3)
        self.assertEqual(context.exception, NegativePopulationError('discrete_adjustment', 'specie_3', 2, -3))

        s2 = DynamicSpecie('specie_3', self.random_state, 3)
        self.assertEqual(s2.get_population(1), 3)

        with self.assertRaises(NegativePopulationError) as context:
            s2.discrete_adjustment(1, -4)
        self.assertEqual(context.exception, NegativePopulationError('discrete_adjustment', 'specie_3', 3, -4))

        pop = 2
        ds_hybrid_1 = DynamicSpecie('ds_hybrid_1', self.random_state, pop, modeled_continuously=True)
        time_0 = 0
        ds_hybrid_1.continuous_adjustment(time_0, -2)
        with self.assertRaises(NegativePopulationError) as context:
            time_1 = 2
            ds_hybrid_1.get_population(time_1)
        self.assertEqual(context.exception, NegativePopulationError('get_population', 'ds_hybrid_1',
            pop, -4.0, time_1 - time_0))

        ds_hybrid_2 = DynamicSpecie('ds_hybrid_2', self.random_state, 2, modeled_continuously=True)
        time = 0
        ds_hybrid_2.continuous_adjustment(time, -3)
        with self.assertRaisesRegexp(NegativePopulationError,
            "continuous_adjustment: population_slope:.*, continuous_time:.*: negative population predicted"):
            time = 1
            ds_hybrid_2.continuous_adjustment(time, 0)

    def test_stochastic_rounding(self):
        s1 = DynamicSpecie('specie', self.random_state, 10.5, modeled_continuously=True)

        samples = 1000
        for i in range(samples):
            pop = s1.get_population(0)
            self.assertTrue(pop in [10, 11])

        mean = np.mean([s1.get_population(0) for i in range(samples) ])
        min = 10 + binom.ppf(0.01, n=samples, p=0.5) / samples
        max = 10 + binom.ppf(0.99, n=samples, p=0.5) / samples
        self.assertTrue(min <= mean <= max)

        s1 = DynamicSpecie('specie', self.random_state, 10.5, modeled_continuously=True)
        s1.continuous_adjustment(1, 0.25)
        for i in range(samples):
            self.assertEqual(s1.get_population(3), 11.0)

    def test_history(self):
        pop = 10
        ds = DynamicSpecie('s', self.random_state, pop, modeled_continuously=True, record_history=True)
        slope = -2
        ds.continuous_adjustment(1, slope)
        discrete_adjustment = 3
        ds.discrete_adjustment(2, discrete_adjustment)
        HistoryRecord = DynamicSpecie.HistoryRecord
        Operation = DynamicSpecie.Operation
        expected_history = [
            HistoryRecord(0, Operation['initialize'], pop),
            HistoryRecord(1, Operation['continuous_adjustment'], slope),
            HistoryRecord(2, Operation['discrete_adjustment'], discrete_adjustment)
        ]
        self.assertEqual(ds.get_history(), expected_history)

        ds = DynamicSpecie('s', self.random_state, 0)
        with self.assertRaisesRegexp(SpeciesPopulationError, 'history not recorded'):
            ds.get_history()


class MockSimulationTestingObject(MockSimulationObject):

    def send_initial_events(self): pass

    def get_state(self):
        return 'object state to be provided'

    def send_debugging_events(self, species_pop_sim_obj, update_time, update_message, update_msg_body,
        get_pop_time, get_pop_msg_body):
        self.send_event(update_time, species_pop_sim_obj, update_msg_body)
        self.send_event(get_pop_time, species_pop_sim_obj, get_pop_msg_body)

    def handle_GivePopulation_event(self, event):
        """ Perform a unit test on the population of self.specie_id."""

        # event.message is a GivePopulation instance
        the_population = event.message.population
        specie_id = self.kwargs['specie_id']
        expected_value = self.kwargs['expected_value']
        self.test_case.assertEqual(the_population[specie_id], expected_value,
            msg="At event_time {} for specie '{}': the correct population "
                "is {} but the actual population is {}.".format(
                event.event_time, specie_id, expected_value, the_population[specie_id]))

    def handle_GiveProperty_event(self, event):
        """ Perform a unit test on the mass of a SpeciesPopSimObject"""
        property_name = event.message.property_name
        self.test_case.assertEqual(property_name, distributed_properties.MASS)
        self.test_case.assertEqual(event.message.value, self.kwargs['expected_value'])

    # register the event handler for each type of message received
    event_handlers =[
            (message_types.GivePopulation, handle_GivePopulation_event),
            (message_types.GiveProperty, handle_GiveProperty_event)]

    # register the message types sent
    messages_sent = [message_types.GetPopulation,
            message_types.AdjustPopulationByDiscreteSubmodel,
            message_types.AdjustPopulationByContinuousSubmodel,
            message_types.GetCurrentProperty]

@unittest.skip("skip until 'change' removed from class ContinuousChange and AdjustPopulationByContinuousSubmodel")
class TestSpeciesPopSimObjectWithAnotherSimObject(unittest.TestCase):
    """ Run a simulation with another simulation object to test SpeciesPopSimObject.

    A SpeciesPopSimObject manages the population of one specie, 'x'. A MockSimulationTestingObject sends
    initialization events to SpeciesPopSimObject and compares the 'x's correct population with
    its simulated population.
    """

    def try_update_species_pop_sim_obj(self, specie_id, init_pop, mol_weight, init_population_slope, update_message,
        msg_body, update_time, get_pop_time, expected_value):
        """ Run a simulation that tests an update of a SpeciesPopSimObject by a update_msg_type message.

        initialize simulation:
            create SpeciesPopSimObject object
            create MockSimulationTestingObject with reference to this TestCase and expected population value
            Mock obj sends update_message for time=update_time
            Mock obj sends GetPopulation for time=get_pop_time
        run simulation:
            SpeciesPopSimObject obj processes both messages
            SpeciesPopSimObject obj sends GivePopulation
            Mock obj receives GivePopulation and checks value
        """
        self.simulator = SimulationEngine()

        if get_pop_time<=update_time:
            raise SpeciesPopulationError('get_pop_time<=update_time')
        species_pop_sim_obj = SpeciesPopSimObject('test_name',
            {specie_id:init_pop}, {specie_id:mol_weight}, initial_population_slopes={specie_id:init_population_slope})
        mock_obj = MockSimulationTestingObject('mock_name', self,
            specie_id=specie_id, expected_value=expected_value)
        self.simulator.add_objects([species_pop_sim_obj, mock_obj])
        mock_obj.send_debugging_events(species_pop_sim_obj, update_time, update_message, msg_body,
            get_pop_time, message_types.GetPopulation({specie_id}))
        self.simulator.initialize()

        self.assertEqual(self.simulator.simulate(get_pop_time+1), 3)

    def test_message_types(self):
        """ Test both discrete and continuous updates, with a range of population & population_slope values"""
        s_id = 's'
        update_adjustment = +5
        get_pop_time = 4
        for s_init_pop in range(3, 7, 2):
            for s_init_population_slope in range(-1, 2):
                for update_time in range(1, 4):

                    self.try_update_species_pop_sim_obj(s_id, s_init_pop, 0, s_init_population_slope,
                        message_types.AdjustPopulationByDiscreteSubmodel,
                        message_types.AdjustPopulationByDiscreteSubmodel({s_id:update_adjustment}),
                        update_time, get_pop_time,
                        s_init_pop + update_adjustment + get_pop_time*s_init_population_slope)

        """
        Test AdjustPopulationByContinuousSubmodel.

        # TODO: IMPT: Delete this and the related code
        Note that the expected_value does not include a term for update_time*s_init_population_slope. This is
        deliberately ignored by `wc_sim.multialgorithm.species_populations.DynamicSpecie()` because it is
        assumed that an adjustment by a continuous submodel will incorporate the population_slope predicted by
        the previous iteration of that submodel.
        """
        for s_init_pop in range(3, 8, 2):
            for s_init_population_slope in range(-1, 2):
                for update_time in range(1, 4):
                    for updated_population_slope in range(-1, 2):
                        self.try_update_species_pop_sim_obj(s_id, s_init_pop, 0, s_init_population_slope,
                            message_types.AdjustPopulationByContinuousSubmodel,
                            message_types.AdjustPopulationByContinuousSubmodel({s_id:
                                message_types.ContinuousChange(update_adjustment, updated_population_slope)}),
                            update_time, get_pop_time,
                            s_init_pop + update_adjustment +
                                (get_pop_time-update_time)*updated_population_slope)


class InitMsg1(SimulationMessage): pass


class TestSpeciesPopSimObject(unittest.TestCase):

    def setUp(self):
        self.simulator = SimulationEngine()
        self.species_ids = 's1 s2 s3'.split()
        self.initial_population = dict(zip(self.species_ids, range(3)))
        self.molecular_weight = dict(zip(self.species_ids, [10]*3))
        self.test_species_pop_sim_obj = SpeciesPopSimObject('test_name', self.initial_population,
            self.molecular_weight)
        self.simulator.add_object(self.test_species_pop_sim_obj)

    def test_init(self):
        for s in self.initial_population.keys():
            self.assertEqual(self.test_species_pop_sim_obj.read_one(0,s), self.initial_population[s])

    def test_invalid_event_types(self):

        with self.assertRaises(SimulatorError) as context:
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj, InitMsg1())
        self.assertIn("'wc_sim.multialgorithm.species_populations.SpeciesPopSimObject' simulation "
            "objects not registered to send", str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj,
                message_types.GivePopulation(7))
        self.assertIn("'wc_sim.multialgorithm.species_populations.SpeciesPopSimObject' simulation "
            "objects not registered to receive", str(context.exception))
