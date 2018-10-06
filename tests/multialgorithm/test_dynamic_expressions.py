"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import warnings
from math import log
import re
import timeit
from scipy.constants import Avogadro

from wc_lang.expression_utils import TokCodes
import wc_lang
from wc_lang import (Model, SpeciesType, Compartment, Species, Parameter, Function, StopCondition,
    FunctionExpression, StopConditionExpression, Observable, ObjectiveFunction, RateLawEquation,
    ExpressionMethods, Concentration, ConcentrationUnit)
from wc_sim.multialgorithm.dynamic_expressions import (DynamicComponent, SimTokCodes, WcSimToken,
    DynamicExpression, DynamicParameter, DynamicFunction, DynamicStopCondition, DynamicObservable,
    DynamicRateLaw, DynamicReaction, DynamicSpecies, MassActionKinetics, WC_LANG_MODEL_TO_DYNAMIC_MODEL)
from wc_sim.multialgorithm.species_populations import MakeTestLSP
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_components import DynamicModel
from wc_sim.multialgorithm.make_models import MakeModels
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation


class TestDynamicExpression(unittest.TestCase):

    def make_objects(self):
        self.model = model = Model()
        self.objects = objects = {
            Observable: {},
            Parameter: {},
            Function: {},
            StopCondition: {}
        }
        self.param_value = 4
        objects[Parameter]['param'] = param = model.parameters.create(id='param', value=self.param_value,
            units='dimensionless')
        model.parameters.create(id='fractionDryWeight', value=0.3, units='dimensionless')

        self.fun_expr = expr = 'param - 2 + max(param, 10)'
        fun1 = ExpressionMethods.make_obj(model, Function, 'fun1', expr, objects)
        fun2 = ExpressionMethods.make_obj(model, Function, 'fun2', 'log(2) - param', objects)

        return model, param, fun1, fun2

    def setUp(self):
        self.init_pop = {}
        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.model, self.parameter, self.fun1, self.fun2 = self.make_objects()

        self.dynamic_model = DynamicModel(self.model, self.local_species_population, {})

        # create a DynamicParameter and a DynamicFunction
        self.dynamic_objects = dynamic_objects = {}
        dynamic_objects[self.parameter] = DynamicParameter(self.dynamic_model, self.local_species_population,
            self.parameter, self.parameter.value)

        for fun in [self.fun1, self.fun2]:
            dynamic_objects[fun] = DynamicFunction(self.dynamic_model, self.local_species_population,
                fun, fun.expression.analyzed_expr)

    def test_simple_dynamic_expressions(self):
        for dyn_obj in self.dynamic_objects.values():
            cls = dyn_obj.__class__
            self.assertEqual(DynamicExpression.dynamic_components[cls][dyn_obj.id], dyn_obj)

        expected_fun1_wc_sim_tokens = [
            WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
            WcSimToken(SimTokCodes.other, '-2+max('),
            WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
            WcSimToken(SimTokCodes.other, ',10)'),
        ]
        expected_fun1_expr_substring = ['', '-2+max(', '', ',10)']
        expected_fun1_local_ns_key = 'max'
        param_val = str(self.param_value)
        expected_fun1_value = eval(self.fun_expr.replace('param', param_val))

        dynamic_expression = self.dynamic_objects[self.fun1]
        dynamic_expression.prepare()
        self.assertEqual(expected_fun1_wc_sim_tokens, dynamic_expression.wc_sim_tokens)
        self.assertEqual(expected_fun1_expr_substring, dynamic_expression.expr_substrings)
        self.assertTrue(expected_fun1_local_ns_key in dynamic_expression.local_ns)
        self.assertEqual(expected_fun1_value, dynamic_expression.eval(0))
        self.assertIn( "id: {}".format(dynamic_expression.id), str(dynamic_expression))
        self.assertIn( "type: {}".format(dynamic_expression.__class__.__name__),
            str(dynamic_expression))
        self.assertIn( "expression: {}".format(dynamic_expression.expression), str(dynamic_expression))

        dynamic_expression = self.dynamic_objects[self.fun2]
        dynamic_expression.prepare()
        expected_fun2_wc_sim_tokens = [ # for 'log(2) - param'
            WcSimToken(SimTokCodes.other, 'log(2)-'),
            WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
        ]
        self.assertEqual(expected_fun2_wc_sim_tokens, dynamic_expression.wc_sim_tokens)

    def test_dynamic_expression_errors(self):
        # remove the Function's tokenized result
        self.fun1.expression.analyzed_expr.wc_tokens = []
        with self.assertRaisesRegexp(MultialgorithmError, "wc_tokens cannot be empty - ensure that '.*' is valid"):
            DynamicFunction(self.dynamic_model, self.local_species_population,
                self.fun1, self.fun1.expression.analyzed_expr)

        expr = 'max(1) - 2'
        fun = ExpressionMethods.make_obj(self.model, Function, 'fun', expr, {}, allow_invalid_objects=True)
        dynamic_function = DynamicFunction(self.dynamic_model, self.local_species_population,
            fun, fun.expression.analyzed_expr)
        dynamic_function.prepare()
        with self.assertRaisesRegexp(MultialgorithmError, re.escape("eval of '{}' raises".format(expr))):
            dynamic_function.eval(1)

        # prepare a dynamic function which refers to a dynamic parameter that doesn't exist
        param_id = 'param_2'
        self.objects[Parameter][param_id] = Parameter()
        fun_id = 'fun_x'
        fun = ExpressionMethods.make_obj(self.model, Function, fun_id, param_id, self.objects)
        dynamic_function = DynamicFunction(self.dynamic_model, self.local_species_population,
            fun, fun.expression.analyzed_expr)
        with self.assertRaisesRegexp(MultialgorithmError,
            "DynamicParameter '{}' must be created before DynamicFunction '{}' can be prepared".format(
            param_id, fun_id)):
            dynamic_function.prepare()

    def test_get_dynamic_model_type(self):

        self.assertEqual(DynamicExpression.get_dynamic_model_type(Function), DynamicFunction)
        with self.assertRaisesRegexp(MultialgorithmError, "model class of type 'FunctionExpression' not found"):
            DynamicExpression.get_dynamic_model_type(FunctionExpression)

        self.assertEqual(DynamicExpression.get_dynamic_model_type(self.fun1), DynamicFunction)
        expr_model_obj, _ = ExpressionMethods.make_expression_obj(Function, '', {})
        with self.assertRaisesRegexp(MultialgorithmError, "model of type 'FunctionExpression' not found"):
            DynamicExpression.get_dynamic_model_type(expr_model_obj)

        self.assertEqual(DynamicExpression.get_dynamic_model_type('Function'), DynamicFunction)
        with self.assertRaisesRegexp(MultialgorithmError, "model type 'NoSuchModel' not defined"):
            DynamicExpression.get_dynamic_model_type('NoSuchModel')
        with self.assertRaisesRegexp(MultialgorithmError, "model type '3' has wrong type"):
            DynamicExpression.get_dynamic_model_type(3)

    def test_get_dynamic_component(self):
        with self.assertRaisesRegexp(MultialgorithmError,
            "model type .* not in DynamicExpression.dynamic_components"):
            DynamicExpression.get_dynamic_component(StopCondition, '')

        with self.assertRaisesRegexp(MultialgorithmError,
            "model type .* with id='.*' not in DynamicExpression.dynamic_components"):
            DynamicExpression.get_dynamic_component(Parameter, 'no_such_id')


class TestAllDynamicExpressionTypes(unittest.TestCase):

    def setUp(self):
        self.objects = objects = {
            Parameter: {},
            Function: {},
            StopCondition: {},
            Observable: {},
            Species: {}
        }

        self.model = model = Model()
        species_types = {}
        st_ids = ['a', 'b']
        for id in st_ids:
            species_types[id] = model.species_types.create(id=id)
        compartments = {}
        comp_ids = ['c1', 'c2']
        for id in comp_ids:
            compartments[id] = model.compartments.create(id=id)
        submodels = {}
        for sm_id, c_id in zip(['submodel1', 'submodel2'], comp_ids):
            submodels[id] = model.submodels.create(id=id, compartment=compartments[c_id])

        for c_id, st_id in zip(comp_ids, st_ids):
            specie = compartments[c_id].species.create(species_type=species_types[st_id])
            objects[Species][specie.get_id()] = specie
            Concentration(species=specie, value=0, units=ConcentrationUnit.M)

        self.init_pop = {'a[c1]': 10, 'b[c2]': 20}

        # map wc_lang object -> expected value
        self.expected_values = expected_values = {}
        param_value = 4
        objects[Parameter]['param'] = param = model.parameters.create(id='param', value=param_value,
            units='dimensionless')
        expected_values[param] = param_value

        # (wc_lang model type, expression, expected value)
        wc_lang_obj_specs = [
            # just reference param:
            (Function, 'param - 2 + max(param, 10)', 12),
            (StopCondition, '10 < 2*log10(100) + 2*param', True),
            # reference other model types:
            (Observable, 'a[c1]', 10),
            (Observable, '2*a[c1] - b[c2]', 0),
            (Function, 'observable_1 + min(observable_2, 10)' , 10),
            (StopCondition, 'observable_1 < param + function_1()', True),
            # reference same model type:
            (Observable, '3*observable_1 + b[c2]', 50),
            (Function, '2*function_2()', 20),
            (Function, '3*observable_1 + function_1()', 42)
        ]

        self.expression_models = expression_models = [Function, StopCondition, Observable]
        last_ids = {wc_lang_type:0 for wc_lang_type in expression_models}
        def make_id(wc_lang_type):
            last_ids[wc_lang_type] += 1
            return "{}_{}".format(wc_lang_type.__name__.lower(), last_ids[wc_lang_type])

        # create wc_lang models
        for wc_lang_model_type, expr, expected_value in wc_lang_obj_specs:
            obj_id = make_id(wc_lang_model_type)
            wc_lang_obj = ExpressionMethods.make_obj(model, wc_lang_model_type, obj_id, expr, objects)
            objects[wc_lang_model_type][obj_id] = wc_lang_obj
            expected_values[wc_lang_obj.id] = expected_value

        # needed for simulation:
        model.parameters.create(id='fractionDryWeight', value=0.3, units='dimensionless')

        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.dynamic_model = DynamicModel(self.model, self.local_species_population, {})

    def test_all_dynamic_expressions(self):

        # check computed value and measure performance of all test Dynamic objects
        number = 1000
        print()
        print("Execution time of Dynamic expressions, averaged over {} evals:".format(number))
        for dynamic_obj_dict in [self.dynamic_model.dynamic_observables,
            self.dynamic_model.dynamic_functions, self.dynamic_model.dynamic_stop_conditions]:
            for id, dynamic_expression in dynamic_obj_dict.items():
                self.assertEqual(self.expected_values[id], dynamic_expression.eval(0))
                eval_time = timeit.timeit(stmt='dynamic_expression.eval(0)', number=number,
                    globals=locals())
                print("{:.2f} usec/eval of {} {} '{}'".format(eval_time*1E6/number,
                    dynamic_expression.__class__.__name__, dynamic_expression.id, dynamic_expression.expression))

def create_test_model(test_case):

        test_case.model = mdl = Model()

        mdl.parameters.create(id='fractionDryWeight', value=0.3, units='dimensionless')
        mdl.parameters.create(id='param_1', value=1.5, units='dimensionless')
        mdl.parameters.create(id='param_2', value=2.5, units='dimensionless')

        test_case.comp = comp = mdl.compartments.create(id='comp_id', name='compartment 0',
            initial_volume=1E-20)

        test_case.extra_comp = extra_comp = mdl.compartments.create(id='extra_comp_id',
            name='compartment extra_comp', initial_volume=1E-20)

        test_case.species = species = []
        species_types = []
        test_case.copy_num = 1E6
        for i in range(10):
            # these species types and species correspond to those created by MakeTestLSP
            spec_type = mdl.species_types.create(id='specie_{}'.format(i),
                name='species type {}'.format(i), molecular_weight=1)
            species_types.append(spec_type)
            spec = Species(species_type=spec_type, compartment=comp)
            Concentration(species=spec, value=test_case.copy_num, units=ConcentrationUnit.molecules)
            species.append(spec)

        # make a species type & species in a different compartment
        test_case.species_in_extra_comp = Species(species_type=species_types[0], compartment=extra_comp)
        Concentration(species=test_case.species_in_extra_comp, value=test_case.copy_num,
            units=ConcentrationUnit.molecules)

        test_case.submdl_0 = submdl_0 = mdl.submodels.create(id='submdl_0', compartment=comp)

        test_case.rxn_0 = rxn_0 = submdl_0.reactions.create(id='order_0')
        equation = RateLawEquation(expression='5')
        test_case.rate_law_0 = rxn_0.rate_laws.create(equation=equation)

        test_case.rxn_1 = rxn_1 = submdl_0.reactions.create(id='order_1')
        rxn_1.participants.create(species=species[0], coefficient=-1)
        equation = RateLawEquation(expression='6')
        test_case.rate_law_1 = rxn_1.rate_laws.create(equation=equation)

        test_case.rxn_2 = rxn_2 = submdl_0.reactions.create(id='order_2_distinct')
        rxn_2.participants.create(species=species[1], coefficient=-1)
        rxn_2.participants.create(species=species[2], coefficient=-1)
        rxn_2.participants.create(species=species[3], coefficient=1)
        equation = RateLawEquation(expression='7')
        test_case.rate_law_2 = rxn_2.rate_laws.create(equation=equation)

        test_case.rxn_3 = rxn_3 = submdl_0.reactions.create(id='order_2_same')
        rxn_3.participants.create(species=species[3], coefficient=-2)
        equation = RateLawEquation(expression='8')
        test_case.rate_law_3 = rxn_3.rate_laws.create(equation=equation)

        test_case.rxn_4 = rxn_4 = submdl_0.reactions.create(id='order_3')
        rxn_4.participants.create(species=species[4], coefficient=-2)
        rxn_4.participants.create(species=species[5], coefficient=-1)
        rxn_4.participants.create(species=species[6], coefficient=0)
        rxn_4.participants.create(species=species[7], coefficient=2)
        equation = RateLawEquation(expression='5')
        test_case.rate_law_4 = rxn_4.rate_laws.create(equation=equation)

        test_case.rxn_5 = rxn_5 = submdl_0.reactions.create(id='has_k_cat')
        test_case.rate_law_5 = rxn_5.rate_laws.create(equation=equation, k_cat=3)

        test_case.rxn_6 = rxn_6 = submdl_0.reactions.create(id='has_k_m')
        test_case.rate_law_6 = rxn_6.rate_laws.create(equation=equation, k_m=3)

        test_case.rxn_8 = rxn_8 = submdl_0.reactions.create(id='not_MA_or_MM')
        equation = RateLawEquation(expression='{}'.format(species[0].get_id()))
        rxn_8.participants.create(species=species[0], coefficient=-1)
        rxn_8.participants.create(species=species[2], coefficient=2)
        equation = RateLawEquation(
            expression='2*{}'.format(species[0].get_id()),
            modifiers=[species[0]])
        test_case.rate_law_8 = rxn_8.rate_laws.create(equation=equation)

        test_case.rxn_9 = rxn_9 = submdl_0.reactions.create(id='coefficient_0.5')
        rxn_9.participants.create(species=species[7], coefficient=-0.5)
        rxn_9.participants.create(species=species[8], coefficient=-1.5)
        equation = RateLawEquation(expression='9')
        test_case.rate_law_9 = rxn_9.rate_laws.create(equation=equation)

        test_case.rxn_10 = rxn_10 = submdl_0.reactions.create(id='Michaelis–Menten_reaction')
        rxn_10.participants.create(species=species[0], coefficient=-2)
        rxn_10.participants.create(species=species[1], coefficient=-3)
        rxn_10.participants.create(species=species[2], coefficient=2)
        equation = RateLawEquation(
            expression='k_cat * {0} / (k_m + {0})'.format(species[3].get_id()),
            modifiers=[species[3]])
        test_case.rate_law_10 = rxn_10.rate_laws.create(equation=equation, k_cat=2, k_m=1)

        # valid_functions = (ceil, floor, exp, pow, log, log10, min, max)
        test_case.rxn_11 = rxn_11 = submdl_0.reactions.create(id='params_and_funcs')
        equation = RateLawEquation(expression='max(param_1, 1) + floor(param_2)')
        test_case.rate_law_11 = rxn_11.rate_laws.create(equation=equation, k_cat=2, k_m=1)
        test_case.rate_law_11.value = 3.5

        multialgorithm_simulation = MultialgorithmSimulation(test_case.model, {})
        _, test_case.dynamic_model = multialgorithm_simulation.build_simulation()
        test_case.dynamic_submodel = multialgorithm_simulation.simulation_submodels[0]
        test_case.species_pop = multialgorithm_simulation.local_species_population
        test_case.dynamic_compartment = multialgorithm_simulation.dynamic_compartments[comp.id]


class TestDynamicComponent(unittest.TestCase):

    def setUp(self):
        create_test_model(self)

    def test_dynamic_component(self):
        dynamic_component1 = DynamicComponent(self.dynamic_model, self.species_pop, self.rate_law_2)
        self.assertIn(self.rate_law_2.reaction.id, str(dynamic_component1))
        dynamic_component2 = DynamicComponent(self.dynamic_model, self.species_pop, self.model.parameters[0])
        self.assertIn(self.model.parameters[0].id, str(dynamic_component2))


class TestMassActionKinetics(unittest.TestCase):

    def setUp(self):
        create_test_model(self)

    def test_init_mass_action_kinetics(self):
        mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
            self.dynamic_compartment, self.rate_law_2)
        self.assertEqual(mass_action_kinetics.id, self.rxn_2.id)
        self.assertEqual(mass_action_kinetics.order, 2)
        self.assertEqual(mass_action_kinetics.rate_constant, 7)
        self.assertEqual(mass_action_kinetics.dynamic_compartment.id, self.comp.id)
        self.assertEqual([d.id for d in mass_action_kinetics.reactants],
            [s.get_id() for s in self.species[1:3]])
        self.assertEqual(mass_action_kinetics.reactant_coefficients, [1, 1])
        with self.assertRaisesRegexp(ValueError, 'not a mass action rate law, as equation uses modifiers'):
            MassActionKinetics(self.dynamic_model, self.species_pop, self.dynamic_compartment,
                self.rate_law_8)
        mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
            self.dynamic_compartment, self.rate_law_11)
        self.assertEqual(mass_action_kinetics.rate_constant, self.rate_law_11.value)

    def test_mass_action_rate_law_asserts(self):
        mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
            self.dynamic_compartment, self.rate_law_4)
        with self.assertRaisesRegexp(AssertionError, "reaction 'order_3': 2 < self.order \(3\)"):
            mass_action_kinetics.calc_mass_action_rate(0)
        mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
            self.dynamic_compartment, self.rate_law_9)
        with self.assertRaisesRegexp(AssertionError,
            "reaction '{}': self.reactant_coefficients\[0\] .* not 1 or 2".format(self.rxn_9.id)):
            mass_action_kinetics.calc_mass_action_rate(0)

    def test_calc_mass_action_rate(self):
        vol_avo = self.dynamic_compartment.volume()*Avogadro
        expected_rates = [  # (rate law, expected rate)
            (self.rate_law_0, 5*vol_avo),
            (self.rate_law_1, 6*self.copy_num),
            (self.rate_law_2, 7*self.copy_num*self.copy_num/vol_avo),
            (self.rate_law_3, 8*self.copy_num*(self.copy_num-1)/(2*vol_avo)),
        ]
        for rate_law, expected_rate in expected_rates:
            mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
                self.dynamic_compartment, rate_law)
            self.assertEqual(mass_action_kinetics.calc_mass_action_rate(0), expected_rate)

    def test_is_mass_action_rate_law(self):
        for mass_action_rate_law in [self.rate_law_0, self.rate_law_1, self.rate_law_2, self.rate_law_3]:
            self.assertTrue(MassActionKinetics.is_mass_action_rate_law(mass_action_rate_law))
        for non_mass_action_rate_law in [self.rate_law_4, self.rate_law_5, self.rate_law_6,
            self.rate_law_8]:
            self.assertFalse(MassActionKinetics.is_mass_action_rate_law(non_mass_action_rate_law))

        rxn = self.submdl_0.reactions.create(id='no_equation')
        rate_law = rxn.rate_laws.create()
        self.assertFalse(MassActionKinetics.is_mass_action_rate_law(rate_law))

    def test_molecularity(self):
        expected_molecularities = [  # (reaction, expected molecularity)
            (self.rxn_0, 0), (self.rxn_1, 1), (self.rxn_2, 2), (self.rxn_3, 2), (self.rxn_4, 3)
        ]
        for rxn, expected_molecularity in expected_molecularities:
            self.assertEqual(MassActionKinetics.molecularity(rxn), expected_molecularity)

    def test_str(self):
        mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
            self.dynamic_compartment, self.rate_law_3)
        self.assertIn(self.rxn_3.id, str(mass_action_kinetics))
        self.assertIn(self.species[3].get_id(), str(mass_action_kinetics))
        self.assertIn(self.comp.id, str(mass_action_kinetics))

    def test_mass_action_rate_law_performance(self):

        # check performance of mass action rate laws
        number = 10000
        print()
        print("Execution time of mass action kinetics, averaged over {} evals:".format(number))
        for rate_law in [self.rate_law_0, self.rate_law_1, self.rate_law_2, self.rate_law_3]:
            mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
                self.dynamic_compartment, rate_law)
            eval_time = timeit.timeit(stmt='mass_action_kinetics.calc_mass_action_rate(0)', number=number,
                globals=locals())
            print("{:.2f} usec/eval of {}".format(eval_time*1E6/number, mass_action_kinetics.id))

        print()
        print("Execution time of mass action kinetics, averaged over {} evals, with compartment volume pre-computed:".format(number))
        volume = self.dynamic_compartment.volume()
        for rate_law in [self.rate_law_0, self.rate_law_1, self.rate_law_2, self.rate_law_3]:
            mass_action_kinetics = MassActionKinetics(self.dynamic_model, self.species_pop,
                self.dynamic_compartment, rate_law)
            eval_time = timeit.timeit(stmt='mass_action_kinetics.calc_mass_action_rate(0, compartment_volume=volume)',
                number=number, globals=locals())
            print("{:.2f} usec/eval of {}".format(eval_time*1E6/number, mass_action_kinetics.id))


class TestDynamicRateLaw(unittest.TestCase):

    def setUp(self):
        create_test_model(self)

    def test_reactants_and_modifiers_compartment(self):
        for rxn in [self.rxn_1, self.rxn_2, self.rxn_3, self.rxn_4, self.rxn_5, self.rxn_6]:
            self.assertEqual(rxn.submodel.compartment,
                DynamicRateLaw.verify_reactants_and_modifiers_compartment(rxn))

    def test_reactants_and_modifiers_compartment_exceptions(self):
        rxn = self.submdl_0.reactions.create(id='species_in_diff_compartments')
        rxn.participants.create(species=self.species[0], coefficient=-1)
        rxn.participants.create(species=self.species_in_extra_comp, coefficient=-1)
        equation = RateLawEquation(expression='6')
        rxn.rate_laws.create(equation=equation)
        with self.assertRaisesRegexp(MultialgorithmError,
            "reactants and modifiers of reaction .* reside in multiple compartments:"):
            DynamicRateLaw.verify_reactants_and_modifiers_compartment(rxn)
        with self.assertRaisesRegexp(MultialgorithmError,
            self.species_in_extra_comp.compartment.id):
            DynamicRateLaw.verify_reactants_and_modifiers_compartment(rxn)

        rxn = self.submdl_0.reactions.create(id='species_not_in_submodel_comp')
        rxn.participants.create(species=self.species_in_extra_comp, coefficient=-1)
        equation = RateLawEquation(expression='6')
        rxn.rate_laws.create(equation=equation)
        with self.assertRaisesRegexp(MultialgorithmError,
            "reactants and/or modifiers of reaction.*are stored in.*which is not its submodel's compartment:"):
            DynamicRateLaw.verify_reactants_and_modifiers_compartment(rxn)

    def test_dynamic_rate_law(self):

        dynamic_rate_laws = []
        volume = self.dynamic_compartment.volume()
        # Mass action rate law
        dynamic_rate_law = DynamicRateLaw(self.dynamic_model, self.species_pop, self.dynamic_compartment,
            self.rate_law_1)
        dynamic_rate_laws.append(dynamic_rate_law)
        self.assertEqual(dynamic_rate_law.mass_action, True)
        self.assertTrue(isinstance(dynamic_rate_law.mass_action_rate_law, MassActionKinetics))
        self.assertEqual(dynamic_rate_law.eval(0), 6*self.copy_num)
        self.assertEqual(dynamic_rate_law.eval(0, compartment_volume=volume), 6*self.copy_num)

        # un typed rate law
        dynamic_rate_law = DynamicRateLaw(self.dynamic_model, self.species_pop, self.dynamic_compartment,
            self.rate_law_8)
        dynamic_rate_laws.append(dynamic_rate_law)
        self.assertEqual(dynamic_rate_law.mass_action, False)
        for attr in ['k_m', 'k_cat']:
            self.assertFalse(hasattr(dynamic_rate_law, attr))
        modifier = self.rate_law_8.equation.modifiers[0]
        self.assertIn("concentrations['{}']".format(modifier.get_id()), dynamic_rate_law.transcoded_equation)
        concentrations = self.dynamic_submodel.get_specie_concentrations()
        self.assertEqual(dynamic_rate_law.eval(0, species_concentrations=concentrations),
            2*concentrations[modifier.get_id()])
        self.assertEqual(
            dynamic_rate_law.eval(0, species_concentrations=concentrations, compartment_volume=volume),
            2*concentrations[modifier.get_id()])

        # MM rate law
        dynamic_rate_law = DynamicRateLaw(self.dynamic_model, self.species_pop, self.dynamic_compartment,
            self.rate_law_10)
        dynamic_rate_laws.append(dynamic_rate_law)
        self.assertEqual(dynamic_rate_law.mass_action, False)
        self.assertEqual(dynamic_rate_law.k_cat, self.rate_law_10.k_cat)
        self.assertEqual(dynamic_rate_law.k_m, self.rate_law_10.k_m)
        self.assertTrue(dynamic_rate_law.michaelis_menten)
        enzyme_id = self.rate_law_10.equation.modifiers[0].get_id()
        self.assertIn("concentrations['{}']".format(enzyme_id), dynamic_rate_law.transcoded_equation)
        mm_rate = self.rate_law_10.k_cat*concentrations[enzyme_id] / (self.rate_law_10.k_m+concentrations[enzyme_id])
        self.assertEqual(dynamic_rate_law.eval(0, species_concentrations=concentrations), mm_rate)
        self.assertEqual(
            dynamic_rate_law.eval(0, species_concentrations=concentrations, compartment_volume=volume), mm_rate)

        number = 10000
        print()
        print("Execution time of dynamic rate law, averaged over {} evals, with compartment volume "
            "pre-computed:".format(number))
        for dynamic_rate_law in dynamic_rate_laws:
            call = 'dynamic_rate_law.eval(0, species_concentrations=concentrations, compartment_volume=volume)'
            eval_time = timeit.timeit(stmt=call, number=number, globals=locals())
            description = dynamic_rate_law.id
            if hasattr(dynamic_rate_law, 'transcoded_equation'):
                description = '{}, with equation "{}"'.format(description, dynamic_rate_law.transcoded_equation)
            print("{:.2f} usec/eval of {}".format(eval_time*1E6/number, description))

    def test_str(self):
        dynamic_rate_law = DynamicRateLaw(self.dynamic_model, self.species_pop, self.dynamic_compartment,
            self.rate_law_10)
        self.assertIn(self.rate_law_10.reaction.id, str(dynamic_rate_law))
        self.assertIn(str(self.rate_law_10.k_cat), str(dynamic_rate_law))
        self.assertIn(str(self.rate_law_10.k_m), str(dynamic_rate_law))
        dynamic_rate_law = DynamicRateLaw(self.dynamic_model, self.species_pop, self.dynamic_compartment,
            self.rate_law_0)
        self.assertIn(self.rate_law_0.reaction.id, str(dynamic_rate_law))


class TestDynamicReaction(unittest.TestCase):

    def setUp(self):
        create_test_model(self)

    def test_dynamic_reaction(self):
        dynamic_reaction = DynamicReaction(self.dynamic_model, self.species_pop, self.rxn_1)
        self.assertEqual(dynamic_reaction.id, self.rxn_1.id)
        self.assertEqual(dynamic_reaction.dynamic_rate_law.id, self.rxn_1.id)
        self.assertIn(self.rxn_1.id, str(dynamic_reaction))
