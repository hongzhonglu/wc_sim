""" Test dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-02-07
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest
import warnings
from argparse import Namespace
from scipy.constants import Avogadro
from itertools import chain

from wc_lang.io import Reader
from wc_lang.core import (Model, Submodel, Compartment, Reaction, SpeciesType, Species, SpeciesCoefficient,
    Concentration, ConcentrationUnit, Observable, ExpressionMethods)
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.dynamic_components import DynamicModel, DynamicCompartment, DynamicCompartmentType
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_expressions import DynamicObservable
from wc_sim.multialgorithm.species_populations import MakeTestLSP
from wc_sim.multialgorithm.make_models import MakeModels


class TestDynamicCompartment(unittest.TestCase):

    def setUp(self):
        comp_id = 'comp_id'

        # make a LocalSpeciesPopulation
        self.num_species = 100
        self.species_nums = species_nums = list(range(0, self.num_species))
        self.species_ids = list(map(lambda x: "specie_{}[{}]".format(x, comp_id), species_nums))
        self.all_pops = 1E6
        self.init_populations = dict(zip(self.species_ids, [self.all_pops]*len(species_nums)))
        self.all_m_weights = 50
        self.molecular_weights = dict(zip(self.species_ids, [self.all_m_weights]*len(species_nums)))
        self.local_species_pop = LocalSpeciesPopulation('test', self.init_populations, self.molecular_weights)

        # make a DynamicCompartment
        self.initial_volume=1E-17
        self.compartment = Compartment(id=comp_id, name='name', initial_volume=self.initial_volume)

    def test_dynamic_compartment_exceptions(self):

        # invalid molecular weights
        compartment = Compartment(id='id', name='name', initial_volume=1)
        invalid_molecular_weights = dict(zip(self.species_ids, [-1]*len(self.species_nums)))
        local_species_pop_w_invalid_mws = LocalSpeciesPopulation('invalid', self.init_populations,
            invalid_molecular_weights)
        with self.assertRaisesRegexp(MultialgorithmError, 'all species types must have positive molecular weights'):
            DynamicCompartment(compartment, local_species_pop_w_invalid_mws, self.species_ids)

        # density is None and invalid init_volume
        for init_volume in [float('nan'), -2, 0, 'x']:
            bad_compartment = Compartment(id='id', name='name', initial_volume=init_volume)
            with self.assertRaisesRegexp(MultialgorithmError,
                "in a biochemical .* init_volume must be a positive real number.*is '{}'".format(init_volume)):
                DynamicCompartment(bad_compartment, self.local_species_pop, self.species_ids)

        # density is None and invalid mass
        empty_local_species_pop = LocalSpeciesPopulation('empty population', {}, {})
        compartment = Compartment(id='id', name='name', initial_volume=1E-20)
        with self.assertRaisesRegexp(MultialgorithmError,
            "initial mass must be a positive real number but it is '0.0'"):
            DynamicCompartment(compartment, empty_local_species_pop)

        # density is provided, but is not a positive real number
        with self.assertRaisesRegexp(MultialgorithmError,
            "density, if provided, must be a positive real number but it is '-1'"):
            DynamicCompartment(compartment, self.local_species_pop, density=-1)

        # abstract DynamicCompartmentType, with bad initial_volume
        for init_volume in [float('nan'), -2, 0, 'x']:
            bad_compartment = Compartment(id='id', name='name', initial_volume=init_volume)
            with self.assertRaisesRegexp(MultialgorithmError,
                "in an abstract .* init_volume must be a positive real number.*is '{}'".format(init_volume)):
                DynamicCompartment(bad_compartment, self.local_species_pop,
                    compartment_type=DynamicCompartmentType.abstract)

        # invalid compartment_type
        invalid_compartment_type = 'foo'
        with self.assertRaisesRegexp(AssertionError,
            "invalid compartment_type: '{}'".format(invalid_compartment_type)):
            DynamicCompartment(compartment, self.local_species_pop, compartment_type=invalid_compartment_type)

    def test_biochemical_dynamic_compartment(self):
        dynamic_compartment = DynamicCompartment(self.compartment, self.local_species_pop)
        self.assertEqual(dynamic_compartment.constant_density,
            self.local_species_pop.compartmental_mass(self.compartment.id)/self.compartment.initial_volume)
        self.assertEqual(dynamic_compartment.mass(), self.local_species_pop.compartmental_mass(self.compartment.id))
        self.assertEqual(dynamic_compartment.volume(), self.compartment.initial_volume)
        self.assertEqual(dynamic_compartment.density(), dynamic_compartment.constant_density)

        # test str()
        self.assertIn(dynamic_compartment.id, str(dynamic_compartment))
        self.assertIn("Compartment type: biochemical", str(dynamic_compartment))
        self.assertIn("Fold change volume: 1.0", str(dynamic_compartment))
        estimated_mass = self.num_species*self.all_pops*self.all_m_weights/Avogadro
        self.assertAlmostEqual(dynamic_compartment.mass(), estimated_mass)
        estimated_density = estimated_mass/self.initial_volume
        self.assertAlmostEqual(dynamic_compartment.density(), estimated_density)

        # self.compartment containing just the first element of self.species_ids
        dynamic_compartment = DynamicCompartment(self.compartment, self.local_species_pop, self.species_ids[:1])
        estimated_mass = self.all_pops*self.all_m_weights/Avogadro
        self.assertAlmostEqual(dynamic_compartment.mass(), estimated_mass)

        # explicit density
        density = 1
        empty_local_species_pop = LocalSpeciesPopulation('empty population', {}, {})
        dynamic_compartment = DynamicCompartment(self.compartment, empty_local_species_pop, density=density)
        self.assertEqual(dynamic_compartment.constant_density, density)
        self.assertEqual(dynamic_compartment.mass(), 0)
        self.assertEqual(dynamic_compartment.volume(), 0)
        self.assertEqual(dynamic_compartment.density(), density)

        # density with 0<mass generates warning
        density = 1
        with warnings.catch_warnings(record=True) as w:
            DynamicCompartment(self.compartment, self.local_species_pop, density=density)
            self.assertEqual(len(w), 1)
            self.assertIn("providing density when 0<self.mass() may cause unexpected behavior", str(w[-1].message))

    def test_abstract_dynamic_compartment(self):
        dynamic_compartment = DynamicCompartment(self.compartment, self.local_species_pop,
            compartment_type=DynamicCompartmentType.abstract)
        self.assertFalse(hasattr(dynamic_compartment, 'constant_density'))
        self.assertTrue(dynamic_compartment.mass() is None)
        self.assertEqual(dynamic_compartment.volume(), self.compartment.initial_volume)
        self.assertTrue(dynamic_compartment.density() is None)

        # check that volume remains constant
        self.local_species_pop.adjust_discretely(0, {self.species_ids[0]:5})
        self.assertEqual(dynamic_compartment.volume(), self.compartment.initial_volume)

        # test str()
        self.assertIn(dynamic_compartment.id, str(dynamic_compartment))
        self.assertIn("Compartment type: abstract", str(dynamic_compartment))


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    def read_model(self, model_filename):
        # read and initialize a model
        self.model = Reader().run(model_filename, strict=False)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        dynamic_compartments = multialgorithm_simulation.dynamic_compartments
        self.dynamic_model = DynamicModel(self.model, multialgorithm_simulation.local_species_population, dynamic_compartments)

    # TODO(Arthur): move this proportional test to a utility & use it instead of assertAlmostEqual everywhere
    def almost_equal_test(self, a, b, frac_diff=1/100):
        delta = min(a, b) * frac_diff
        self.assertAlmostEqual(a, b, delta=delta)

    def compare_aggregate_states(self, expected_aggregate_state, computed_aggregate_state):
        list_of_nested_keys_to_test = [
            ['cell mass'],
            ['cell volume'],
            ['compartments', 'c', 'mass'],
            ['compartments', 'c', 'volume']
        ]
        for nested_keys_to_test in list_of_nested_keys_to_test:
            expected = expected_aggregate_state
            computed = computed_aggregate_state
            for key in nested_keys_to_test:
                expected = expected[key]
                computed = computed[key]
            self.almost_equal_test(expected, computed)

    # TODO(Arthur): test with multiple compartments
    def test_dynamic_model(self):
        self.read_model(self.MODEL_FILENAME)
        self.assertEqual(self.dynamic_model.fraction_dry_weight, 0.3)

        # expected values computed in tests/multialgorithm/fixtures/test_model_with_mass_computation.xlsx
        self.almost_equal_test(self.dynamic_model.cell_mass(), 8.260E-16)
        self.almost_equal_test(self.dynamic_model.cell_dry_weight(), 2.48E-16)
        expected_aggregate_state = {
            'cell mass': 8.260E-16,
            'cell volume': 4.58E-17,
            'compartments': {'c':
                {'mass': 8.260E-16,
                'name': 'Cell',
                'volume': 4.58E-17}}
        }
        computed_aggregate_state = self.dynamic_model.get_aggregate_state()
        self.compare_aggregate_states(expected_aggregate_state, computed_aggregate_state)

    def test_dry_dynamic_model(self):
        self.read_model(self.DRY_MODEL_FILENAME)
        self.assertEqual(self.dynamic_model.fraction_dry_weight, 0)

        # expected values computed in tests/multialgorithm/fixtures/test_dry_model_with_mass_computation.xlsx
        self.almost_equal_test(self.dynamic_model.cell_mass(), 9.160E-19)
        self.almost_equal_test(self.dynamic_model.cell_dry_weight(), 9.160E-19)
        aggregate_state = self.dynamic_model.get_aggregate_state()
        computed_aggregate_state = self.dynamic_model.get_aggregate_state()
        expected_aggregate_state = {
            'cell mass': 9.160E-19,
            'cell volume': 4.58E-17,
            'compartments': {'c':
                {'mass': 9.160E-19,
                'name': 'Cell',
                'volume': 4.58E-17}}
        }
        self.compare_aggregate_states(expected_aggregate_state, computed_aggregate_state)

    def test_eval_dynamic_observables(self):
        # make a Model
        model = Model()
        comp = model.compartments.create(id='comp_0')
        submodel = model.submodels.create(id='submodel', compartment=comp)
        model.parameters.create(id='fractionDryWeight', value=0.3)

        num_species_types = 10
        species_types = []
        for i in range(num_species_types):
            st = model.species_types.create(id='st_{}'.format(i))
            species_types.append(st)

        species = []
        for st_idx in range(num_species_types):
            specie = comp.species.create(species_type=species_types[st_idx])
            conc = Concentration(species=specie, value=0, units=ConcentrationUnit.M)
            species.append(specie)

        # create some observables
        objects = {
            Species:{},
            Observable:{}
        }
        num_non_dependent_observables = 5
        non_dependent_observables = []
        for i in range(num_non_dependent_observables):
            expr_parts = []
            for j in range(i+1):
                expr_parts.append("{}*{}".format(j, species[j].get_id()))
                objects[Species][species[j].get_id()] = species[j]
            expr = ' + '.join(expr_parts)
            obj = ExpressionMethods.make_obj(model, Observable, 'obs_nd_{}'.format(i), expr, objects)
            self.assertTrue(obj.expression.validate() is None)
            non_dependent_observables.append(obj)

        num_dependent_observables = 4
        dependent_observables = []
        for i in range(num_dependent_observables):
            expr_parts = []
            for j in range(i+1):
                nd_obs_id = 'obs_nd_{}'.format(j)
                expr_parts.append("{}*{}".format(j, nd_obs_id))
                objects[Observable][nd_obs_id] = non_dependent_observables[j]
            expr = ' + '.join(expr_parts)
            obj = ExpressionMethods.make_obj(model, Observable, 'obs_d_{}'.format(i), expr, objects)
            self.assertTrue(obj.expression.validate() is None)
            dependent_observables.append(obj)

        # make a LocalSpeciesPopulation
        init_pop = dict(zip([s.id() for s in species], list(range(num_species_types))))
        lsp = MakeTestLSP(initial_population=init_pop).local_species_pop

        # make a DynamicModel
        dyn_mdl = DynamicModel(model, lsp, {})
        # check that dynamic observables have the right values
        for obs_id, obs_val in dyn_mdl.eval_dynamic_observables(0).items():
            index = int(obs_id.split('_')[-1])
            if 'obs_nd_' in obs_id:
                expected_val = float(sum([i*i for i in range(index+1)]))
                self.assertEqual(expected_val, obs_val)
            elif 'obs_d_' in obs_id:
                expected_val = 0
                for d_index in range(index+1):
                    expected_val += d_index * sum([i*i for i in range(d_index+1)])
                self.assertEqual(expected_val, obs_val)
