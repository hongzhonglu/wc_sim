"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import unittest
import os
from argparse import Namespace
import math
import re
import numpy as np
from scipy.constants import Avogadro

from obj_model import utils
from wc_utils.util.enumerate import CaseInsensitiveEnum
from wc_lang.io import Reader, Writer
from wc_lang.core import (Model, Submodel,  SpeciesType, SpeciesTypeType, Species,
                          Reaction, Observable, Compartment,
                          SpeciesCoefficient, ObservableCoefficient, Parameter,
                          RateLaw, RateLawDirection, RateLawEquation, SubmodelAlgorithm, Concentration,
                          BiomassComponent, BiomassReaction, StopCondition)
from wc_lang.prepare import PrepareModel, CheckModel
from wc_lang.transform import SplitReversibleReactionsTransform
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.config import core as config_core_multialgorithm
config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class TestMultialgorithmSimulation(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        for base_model in [Submodel, Species, SpeciesType]:
            base_model.objects.reset()
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME, strict=False)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)

    def test_molecular_weights_for_species(self):
        multi_alg_sim = self.multialgorithm_simulation
        self.assertEqual(multi_alg_sim.molecular_weights_for_species(set()), {})
        expected = {
            'specie_6[c]':6,
            'H2O[c]':18.0152
        }
        self.assertEqual(multi_alg_sim.molecular_weights_for_species(set(expected.keys())),
            expected)

    def test_partition_species(self):
        self.multialgorithm_simulation.partition_species()
        expected_priv_species = dict(
            submodel_1=['specie_1[e]', 'specie_2[e]', 'specie_1[c]'],
            submodel_2=['specie_4[c]', 'specie_5[c]', 'specie_6[c]']
        )
        self.assertEqual(self.multialgorithm_simulation.private_species, expected_priv_species)
        expected_shared_species = set(['specie_2[c]', 'specie_3[c]', 'H2O[e]', 'H2O[c]'])
        self.assertEqual(self.multialgorithm_simulation.shared_species, expected_shared_species)

    def test_dynamic_compartments(self):
        expected_compartments = dict(
            submodel_1=['c', 'e'],
            submodel_2=['c']
        )
        for submodel_id in ['submodel_1', 'submodel_2']:
            submodel = Submodel.objects.get_one(id=submodel_id)
            submodel_dynamic_compartments = self.multialgorithm_simulation.get_dynamic_compartments(submodel)
            self.assertEqual(set(submodel_dynamic_compartments.keys()), set(expected_compartments[submodel_id]))

    def test_static_methods(self):
        initial_species_population = MultialgorithmSimulation.get_initial_species_pop(self.model)
        specie_wo_init_conc = 'specie_3[c]'
        self.assertEqual(initial_species_population[specie_wo_init_conc], 0)
        self.assertEqual(initial_species_population['specie_2[c]'], initial_species_population['specie_4[c]'])
        for concentration in self.model.get_concentrations():
            self.assertGreater(initial_species_population[concentration.species.id()], 0)

        local_species_population = MultialgorithmSimulation.make_local_species_pop(self.model)
        self.assertEqual(local_species_population.read_one(0, specie_wo_init_conc), 0)

    def test_build_simulation(self):
        self.simulation_engine, _ = self.multialgorithm_simulation.build_simulation()
        self.assertEqual(len(self.simulation_engine.simulation_objects.keys()), 2)


class RateLawType(int, CaseInsensitiveEnum):
    """ Rate law typ """
    constant = 1
    reactant_pop = 2
    product_pop = 3


class TestRunSimulation(unittest.TestCase):

    @staticmethod
    def get_model_type_params(model_type):
        """ Given a model type, generate params for creating the model
        """
        num_species = 0
        result = re.search(r'(\d) species', model_type)
        if result:
            num_species = int(result.group(1))

        num_reactions = 0
        result = re.search(r'(\d) reaction', model_type)
        if result:
            num_reactions = int(result.group(1))

        reversible = False
        if 'pair of symmetrical reactions' in model_type:
            reversible = True
            num_reactions = 1

        rate_law_type = RateLawType.constant
        if 'rates given by reactant population' in model_type:
            rate_law_type = RateLawType.reactant_pop
        if 'rates given by product population' in model_type:
            rate_law_type = RateLawType.product_pop

        return (num_species, num_reactions, reversible, rate_law_type)

    @staticmethod
    def convert_pop_conc(specie_copy_number, vol):
        return specie_copy_number/(vol*Avogadro)

    def make_test_model(self, model_type, specie_copy_number=1000000, init_vol=1E-16):
        """ Create a test model

        Args:
            model_type (:obj:`str`): model type description
            specie_copy_number (:obj:`int`): population of each species in its compartment
        """
        concentration = self.convert_pop_conc(specie_copy_number, init_vol)
        print('concentration', concentration)

        num_species, num_reactions, reversible, rate_law_type = self.get_model_type_params(model_type)
        if (2<num_species or 1<num_reactions or
            (0<num_reactions and num_species==0) or
            (rate_law_type == RateLawType.product_pop and num_species != 2)):
            raise ValueError("invalid combination of num_species ({}), num_reactions ({}), rate_law_type ({})".format(
                num_species, num_reactions, rate_law_type.name))

        # Model
        model = Model(id='test_model', version='0.0.0', wc_lang_version='0.0.1')
        # Compartment
        comp = model.compartments.create(id='c', initial_volume=init_vol)

        # SpeciesTypes, Species and Concentrations
        species = []
        for i in range(num_species):
            spec_type = model.species_types.create(
                id='spec_type_{}'.format(i),
                type=SpeciesTypeType.protein,
                molecular_weight=10)
            spec = comp.species.create(species_type=spec_type)
            species.append(spec)
            Concentration(species=spec, value=concentration)
        # Submodel
        submodel = model.submodels.create(id='test_submodel', algorithm=SubmodelAlgorithm.ssa,
            compartment=comp)

        # Reactions and RateLaws
        if num_species:
            backward_product = forward_reactant = species[0]
            if 1<num_species:
                backward_reactant = forward_product = species[1]

        # ignore modifiers, which aren't used by the simulator
        if num_reactions:
            reaction = submodel.reactions.create(id='test_reaction_1', reversible=reversible)
            reaction.participants.create(species=forward_reactant, coefficient=-1)
            if rate_law_type.name =='constant':
                equation=RateLawEquation(expression='1')
            if rate_law_type.name == 'reactant_pop':
                equation=RateLawEquation(expression=forward_reactant.id())
            if rate_law_type.name == 'product_pop':
                equation=RateLawEquation(expression=forward_product.id())
            reaction.rate_laws.create(direction=RateLawDirection.forward, equation=equation)

            if num_species == 2:
                reaction.participants.create(species=forward_product, coefficient=1)

            if reversible:
                # make backward rate law
                # RateLawEquations identical to the above must be recreated so backreferences work
                if rate_law_type.name == 'constant':
                    equation=RateLawEquation(expression='1')
                if rate_law_type.name == 'reactant_pop':
                    equation=RateLawEquation(expression=backward_reactant.id())
                if rate_law_type.name == 'product_pop':
                    equation=RateLawEquation(expression=backward_product.id())

                rate_law = RateLaw(direction=RateLawDirection.backward, equation=equation)
                reaction.rate_laws.add(rate_law)

        # Parameters
        model.parameters.create(id='fractionDryWeight', value=0.3)
        model.parameters.create(id='carbonExchangeRate', value=12, units='mmol/gDCW/h')
        model.parameters.create(id='nonCarbonExchangeRate', value=20, units='mmol/gDCW/h')

        # prepare & check the model
        SplitReversibleReactionsTransform().run(model)
        PrepareModel(model).run()
        # check model transcodes the rate law expressions
        CheckModel(model).run()
        '''
        model.pprint(max_depth=2)
        # pprint(show=['', '', ]
        if num_reactions:
            model.submodels[0].reactions[0].pprint(max_depth=2)
            for part in model.submodels[0].reactions[0].participants:
                part.pprint(max_depth=1)
            for ratelaw in model.submodels[0].reactions[0].rate_laws:
                ratelaw.pprint(max_depth=2)
        '''

        # create Manager indices
        # TODO(Arthur): should be automated in a finalize() method
        for base_model in [Submodel,  SpeciesType, Reaction, Observable, Compartment, Parameter]:
            base_model.get_manager().insert_all_new()

        self.model = model

    def test_make_test_model(self):
        '''
        Simple SSA model tests:
            no reactions: simulation terminates immediately
            1 species:
                one reaction consume specie, at constant rate: consume all reactant, in time given by rate
            2 species:
                one reaction: convert all reactant into product, in time given by rate
                a pair of symmetrical reactions with constant rates: maintain steady state, on average
                a pair of symmetrical reactions rates given by reactant population: maintain steady state, on average
                a pair of symmetrical reactions rates given by product population: exhaust on species, with equal chance for each species
            ** ring of futile reactions with balanced rates: maintain steady state, on average
        '''
        model_types = ['no reactions',
            '1 species, 1 reaction',
            '2 species, 1 reaction',
            '2 species, a pair of symmetrical reactions with constant rates',
            '2 species, a pair of symmetrical reactions rates given by reactant population',
            '2 species, a pair of symmetrical reactions rates given by product population',
        ]

        # test get_model_type_params
        expected_params_list = [
            (0, 0, False, RateLawType.constant),
            (1, 1, False, RateLawType.constant),
            (2, 1, False, RateLawType.constant),
            (2, 1, True, RateLawType.constant),
            (2, 1, True, RateLawType.reactant_pop),
            (2, 1, True, RateLawType.product_pop)
        ]
        for model_type,expected_params in zip(model_types, expected_params_list):
            params = self.get_model_type_params(model_type)
            self.assertEqual(params, expected_params)

        # test make_test_model
        for model_type in model_types:
            self.make_test_model(model_type)
        '''
            # if necessary, write model to spreadsheet
            file = model_type.replace(' ', '_')
            filename = os.path.join(os.path.dirname(__file__), 'tmp', file+'.xlsx')
            Writer().run(self.model, filename)
            print('wrote model to:', filename)
        '''
        # unittest one of the models made
        self.make_test_model(model_types[4])
        self.assertEqual(self.model.id, 'test_model')
        comp = self.model.compartments[0]
        self.assertEqual(comp.id, 'c')
        species_type_ids = set([st.id for st in self.model.species_types])
        self.assertEqual(species_type_ids, set(['spec_type_0', 'spec_type_1']))
        species_ids = set([s.id() for s in comp.species])
        self.assertEqual(species_ids, set(['spec_type_0[c]', 'spec_type_1[c]']))
        submodel = self.model.submodels[0]
        self.assertEqual(self.model.submodels[0].compartment, comp)

        # reaction was split by SplitReversibleReactionsTransform
        ratelaw_elements = set()
        for r in submodel.reactions:
            self.assertFalse(r.reversible)
            rl = r.rate_laws[0]
            ratelaw_elements.add((rl.direction, rl.equation.expression))
        expected_rate_laws = set([
            # direction, equation expression
            (RateLawDirection.forward, 'spec_type_0[c]'),   # forward
            (RateLawDirection.forward, 'spec_type_1[c]'),   # backward, but reversed
        ])
        self.assertEqual(ratelaw_elements, expected_rate_laws)

        participant_elements = set()
        for r in submodel.reactions:
            r_list = []
            for part in r.participants:
                r_list.append((part.species.id(), part.coefficient))
            participant_elements.add(tuple(sorted(r_list)))
        expected_participants = set([
            # id, coefficient
            tuple(sorted((('spec_type_0[c]', -1), ('spec_type_1[c]',  1)))),    # forward
            tuple(sorted((('spec_type_1[c]', -1), ('spec_type_0[c]',  1)))),    # reversed
        ])
        self.assertEqual(participant_elements, expected_participants)
        self.assertIn('fractionDryWeight', [p.id for p in self.model.get_parameters()])

        # TODO(Arthur): NEXT, run SSA with '1 species, 1 reaction'

    def setUp(self):
        for base_model in [Submodel, Species, SpeciesType]:
            base_model.objects.reset()
        ### make simple model ###
        self.make_test_model('1 species, 1 reaction')
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)

    def test_run_simulation(self):
        self.simulation_engine, _ = self.multialgorithm_simulation.build_simulation()
        for name,simulation_obj in self.simulation_engine.simulation_objects.items():
            print("\n{}: {} event queue:".format(simulation_obj.__class__.__name__, name))
            print(simulation_obj.render_event_queue())
        # self.simulation_engine.initialize()
