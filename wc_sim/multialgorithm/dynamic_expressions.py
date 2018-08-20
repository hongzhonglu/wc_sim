""" Dynamic expressions

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

import re
import os
import warnings
import tempfile
import math
from collections import namedtuple
from scipy.constants import Avogadro

import obj_model
from wc_utils.util.enumerate import CaseInsensitiveEnum
import wc_utils.cache
import wc_lang
from wc_lang import (Species, Reaction, Parameter, StopCondition, Function, Observable,
    ObjectiveFunction, RateLaw, RateLawEquation)
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.expression_utils import TokCodes, RateLawUtils

'''
# TODO:
build:
    integrate into dynamic simulation
    support parameters in rate_law.equation.expression by making it a DynamicExpression
    don't create duplicate DynamicSpecies, or any other Dynamic* object; generalize creation this to get_or_create()
    make all of DynamicSubmodel's methods handle DynamicReactions, & replace self.reactions w self.dynamic_reactions
    think about and support reactants and modifiers in multiple compartments
    in "return RateLawUtils.eval_rate_law(self, ..." raise exception if species_concentrations is None
cleanup
    move dynamic_components to a more convenient place; cannot leave DynamicComponent defined in this file
    jupyter examples
    memoize performance comparison; decide whether to trash or finish implementing direct dependency tracking eval
    clean up memoize cache file?
optimizations
Expression eval design:
    Algorithms:
        evaling expression model types:
            special cases:
                ObjectiveFunction: used by FBA, so express as needed by the FBA solver
                RateLawEquation: needs special considaration of reactant order, intensive vs. extensive, volume, etc.
        evaling other model types used by expressions:
            Reaction and BiomassReaction: flux units in ObjectiveFunction?
    Optimizations:
        evaluate parameters statically at initialization
        use memoization to avoid re-evaluation, if the benefit outweighs the overhead; like this:
            cache_dir = tempfile.mkdtemp()
            cache = wc_utils.cache.Cache(directory=os.path.join(cache_dir, 'cache'))
            @cache.memoize()
            def eval(time):
        fast access to specie counts and concentrations:
            eliminate lookups, extra objects and memory allocation/deallocation
        for maximum speed, don't use eval() -- convert expressions into trees, & use an evaluator that
            can process operators, literals, and Python functions
'''


class SimTokCodes(int, CaseInsensitiveEnum):
    """ Token codes used in WcSimTokens """
    dynamic_expression = 1
    other = 2


# a token in DynamicExpression.wc_tokens
WcSimToken = namedtuple('WcSimToken', 'tok_code, token_string, dynamic_expression')
# make dynamic_expression optional: see https://stackoverflow.com/a/18348004
WcSimToken.__new__.__defaults__ = (None, )
WcSimToken.__doc__ += ': Token in a validated expression'
WcSimToken.tok_code.__doc__ = 'SimTokCodes encoding'
WcSimToken.token_string.__doc__ = "The token's string"
WcSimToken.dynamic_expression.__doc__ = "When tok_code is dynamic_expression, the dynamic_expression instance"


class DynamicComponent(object):
    """ Component of a simulation

    Attributes:
        dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
        local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
        id (:obj:`str`): unique id; dynamic components use the same `id` as the corresponding
            `wc_lang_model`, unless it does not have an id; exceptions:
                * `DynamicRateLaw` uses the `id` of the `RateLaw`'s `reaction`
    """
    def __init__(self, dynamic_model, local_species_population, wc_lang_model):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): a corresponding `wc_lang` `Model`, from which this
                `DynamicComponent` is derived
        """
        self.dynamic_model = dynamic_model
        self.local_species_population = local_species_population
        self.id = wc_lang_model.get_id()
        model_type = DynamicExpression.get_dynamic_model_type(wc_lang_model)
        if model_type not in DynamicExpression.dynamic_components:
            DynamicExpression.dynamic_components[model_type] = {}
        DynamicExpression.dynamic_components[model_type][self.id] = self

    def __str__(self):
        """ Provide a readable representation of this `DynamicComponent`

        Returns:
            :obj:`str`: a readable representation of this `DynamicComponent`
        """
        rv = ['DynamicComponent:']
        rv.append("type: {}".format(self.__class__.__name__))
        rv.append("id: {}".format(self.id))
        return '\n'.join(rv)


class DynamicExpression(DynamicComponent):
    """ Simulation representation of a mathematical expression, based on WcLangExpression

    Attributes:
        expression (:obj:`str`): the expression defined in the `wc_lang` Model
        wc_sim_tokens (:obj:`list` of `WcSimToken`): a tokenized, compressed representation of `expression`
        expr_substrings (:obj:`list` of `str`): strings which are joined to form the string which is 'eval'ed
        local_ns (:obj:`dict`): pre-computed local namespace of functions used in `expression`
    """

    def __init__(self, dynamic_model, local_species_population, wc_lang_model, wc_lang_expression):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): the corresponding `wc_lang` `Model`
            wc_lang_expression (:obj:`WcLangExpression`): an analyzed and validated expression

        Raises:
            :obj:`MultialgorithmError`: if `wc_lang_expression` does not contain an analyzed,
                validated expression
        """

        super().__init__(dynamic_model, local_species_population, wc_lang_model)

        # wc_lang_expression must have been successfully `tokenize`d.
        if not wc_lang_expression.wc_tokens:
            raise MultialgorithmError("wc_tokens cannot be empty - ensure that '{}' is valid".format(
                wc_lang_model))
        # optimization: self.wc_lang_expression will be deleted by prepare()
        self.wc_lang_expression = wc_lang_expression
        self.expression = wc_lang_expression.expression

    def prepare(self):
        """ Prepare this dynamic expression for simulation

        Because they refer to each other, all `DynamicExpression`s must be created before any of them
        are prepared.

        Raises:
            :obj:`MultialgorithmError`: if a Python function used in `wc_lang_expression` does not exist
        """

        # create self.wc_sim_tokens, which contains WcSimTokens that refer to other DynamicExpressions
        self.wc_sim_tokens = []
        # optimization: combine together adjacent wc_token.tok_codes other than wc_lang_obj_id
        next_static_tokens = ''
        function_names = set()
        non_lang_obj_id_tokens = set([TokCodes.math_fun_id, TokCodes.number, TokCodes.op, TokCodes.other])

        i = 0
        while i <  len(self.wc_lang_expression.wc_tokens):
            wc_token = self.wc_lang_expression.wc_tokens[i]
            if wc_token.tok_code == TokCodes.math_fun_id:
                function_names.add(wc_token.token_string)
            if wc_token.tok_code in non_lang_obj_id_tokens:
                next_static_tokens = next_static_tokens + wc_token.token_string
            elif wc_token.tok_code == TokCodes.wc_lang_obj_id:
                if next_static_tokens != '':
                    self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
                    next_static_tokens = ''
                try:
                    dynamic_expression = self.get_dynamic_component(wc_token.model, wc_token.model_id)
                except:
                    raise MultialgorithmError("{} '{}' must be created before {} '{}' can be prepared'".format(
                        DynamicExpression.get_dynamic_model_type(wc_token.model.__class__).__name__, wc_token.model_id,
                        self.__class__.__name__, self.id))
                self.wc_sim_tokens.append(WcSimToken(SimTokCodes.dynamic_expression, wc_token.token_string,
                    dynamic_expression))
            else:   # pragma    no cover
                assert False, "unknown tok_code {} in {}".format(wc_token.tok_code, wc_token)
            if wc_token.tok_code == TokCodes.wc_lang_obj_id and wc_token.model_type == Function:
                # skip past the () syntactic sugar on functions
                i += 2
            # advance to the next token
            i += 1
        if next_static_tokens != '':
            self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
        # optimization: to conserve memory, delete self.wc_lang_expression
        del self.wc_lang_expression

        # optimization: pre-allocate and pre-populate substrings for the expression to eval
        self.expr_substrings = []
        for sim_token in self.wc_sim_tokens:
            if sim_token.tok_code == SimTokCodes.other:
                self.expr_substrings.append(sim_token.token_string)
            else:
                self.expr_substrings.append('')

        # optimization: pre-allocate Python functions in namespace
        self.local_ns = {}
        for func_name in function_names:
            if func_name in globals()['__builtins__']:
                self.local_ns[func_name] = globals()['__builtins__'][func_name]
            elif hasattr(globals()['math'], func_name):
                self.local_ns[func_name] = getattr(globals()['math'], func_name)
            else:   # pragma no cover, because only known functions are allowed in model expressions
                raise MultialgorithmError("loading expression '{}' cannot find function '{}'".format(
                    self.expression, func_name))

    def eval(self, time):
        """ Evaluate this mathematical expression

        Approach:
            * Replace references to related Models in `self.wc_sim_tokens` with their values
            * Join the elements of `self.wc_sim_tokens` into a Python expression
            * `eval` the Python expression

        Args:
            time (:obj:`float`): the current simulation time

        Raises:
            :obj:`MultialgorithmError`: if Python `eval` raises an exception
        """
        assert hasattr(self, 'wc_sim_tokens'), "'{}' must use prepare() before eval()".format(
            self.id)
        for idx, sim_token in enumerate(self.wc_sim_tokens):
            if sim_token.tok_code == SimTokCodes.dynamic_expression:
                self.expr_substrings[idx] = str(sim_token.dynamic_expression.eval(time))
        try:
            return eval(''.join(self.expr_substrings), {}, self.local_ns)
        except BaseException as e:
            raise MultialgorithmError("eval of '{}' raises {}: {}'".format(
                self.expression, type(e).__name__, str(e)))

    dynamic_components = {}

    @staticmethod
    def get_dynamic_model_type(model_type):
        """ Get a simulation's dynamic component type

        Convert to a dynamic component type from a corresponding `wc_lang` Model type, instance or
        string name

        Args:
            model_type (:obj:`obj`): a `wc_lang` Model type represented by a subclass of `obj_model.Model`,
                an instance of `obj_model.Model`, or a string name for a `obj_model.Model`

        Returns:
            :obj:`type`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the corresponding dynamic component type cannot be determined
        """
        if isinstance(model_type, type) and issubclass(model_type, obj_model.Model):
            if model_type in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type]
            raise MultialgorithmError("model class of type '{}' not found".format(model_type.__name__))

        if isinstance(model_type, obj_model.Model):
            if model_type.__class__ in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type.__class__]
            raise MultialgorithmError("model of type '{}' not found".format(model_type.__class__.__name__))

        if isinstance(model_type, str):
            if model_type in globals():
                model_type = globals()[model_type]
                if not isinstance(model_type, str): # avoid infinite recursion # pragma no cover, but hand tested
                    return DynamicExpression.get_dynamic_model_type(model_type)
            raise MultialgorithmError("model type '{}' not defined".format(model_type))
        raise MultialgorithmError("model type '{}' has wrong type".format(model_type))

    @staticmethod
    def get_dynamic_component(model_type, id):
        """ Get a simulation's dynamic component

        Args:
            model_type (:obj:`type`): the subclass of `DynamicComponent` (or `obj_model.Model`) being retrieved
            id (:obj:`str`): the dynamic component's id

        Returns:
            :obj:`DynamicComponent`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the dynamic component cannot be found
        """
        model_type = DynamicExpression.get_dynamic_model_type(model_type)
        if model_type not in DynamicExpression.dynamic_components:
            raise MultialgorithmError("model type '{}' not in DynamicExpression.dynamic_components".format(
                model_type.__name__))
        if id not in DynamicExpression.dynamic_components[model_type]:
            raise MultialgorithmError("model type '{}' with id='{}' not in DynamicExpression.dynamic_components".format(
                model_type.__name__, id))
        # print('model_type, id', model_type, id, '->', DynamicExpression.dynamic_components[model_type][id])
        return DynamicExpression.dynamic_components[model_type][id]

    def __str__(self):
        """ Provide a readable representation of this `DynamicExpression`

        Returns:
            :obj:`str`: a readable representation of this `DynamicExpression`
        """
        rv = ['DynamicExpression:']
        rv.append("type: {}".format(self.__class__.__name__))
        rv.append("id: {}".format(self.id))
        rv.append("expression: {}".format(self.expression))
        return '\n'.join(rv)


class DynamicFunction(DynamicExpression):
    """ The dynamic representation of a `Function`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicStopCondition(DynamicExpression):
    """ The dynamic representation of a `StopCondition`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicObservable(DynamicExpression):
    """ The dynamic representation of an `Observable`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicParameter(DynamicComponent):
    """ The dynamic representation of a `Parameter`
    """

    def __init__(self, dynamic_model, local_species_population, parameter, value):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            parameter (:obj:`wc_lang.core.Parameter`): the corresponding `wc_lang` `Parameter`
            value (:obj:`float`): the parameter's value
        """
        super().__init__(dynamic_model, local_species_population, parameter)
        self.value = value

    def eval(self, time):
        """ Provide the value of this parameter

        Args:
            time (:obj:`float`): the current simulation time; not needed, but included so that all
                dynamic expression models have the same signature for 'eval`
        """
        return self.value


class DynamicSpecies(DynamicComponent):
    """ The dynamic representation of a `Species`
    """

    def __init__(self, dynamic_model, local_species_population, species):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            species (:obj:`wc_lang.core.Species`): the corresponding `wc_lang` `Species`
        """
        super().__init__(dynamic_model, local_species_population, species)
        # Grab a reference to the right Species object used by local_species_population
        self.species_obj = local_species_population._population[species.get_id()]

    def eval(self, time):
        """ Provide the population of this species

        Args:
            time (:obj:`float`): the current simulation time
        """
        return self.species_obj.get_population(time)

    def get_population(self, time):
        """ Provide the population of this species

        Args:
            time (:obj:`float`): the current simulation time
        """
        return self.species_obj.get_population(time)


class DynamicObjectiveFunction(DynamicExpression):
    """ The dynamic representation of an `ObjectiveFunction`
    """

    def __init__(self, *args):
        super().__init__(*args)


class MassActionKinetics(DynamicComponent):
    """ Mass action kinetics for a rate law

    Attributes:
        id (:obj:`str`): reaction id
        order (:obj:`int`): order
        rate_constant (:obj:`float`): the rate law's rate constant
        reactants (:obj:`list` of `DynamicSpecies`): the reactants
        reactant_coefficients (:obj:`list` of `float`): the reactants' reactant_coefficients
        dynamic_compartment (:obj:`DynamicCompartment`): the `DynamicCompartment` that contains
            the reactants in `rate_law.reaction`
    """

    def __init__(self, dynamic_model, local_species_population, dynamic_compartment, rate_law):
        """ Create an instance of MassActionKinetics

        Args:
            rate_law (:obj:`wc_lang.core.RateLaw`): a RateLaw instance
        """
        super().__init__(dynamic_model, local_species_population, rate_law.reaction)
        self.id = rate_law.reaction.id
        self.dynamic_compartment = dynamic_compartment
        self.order = MassActionKinetics.molecularity(rate_law.reaction)
        try:
            self.rate_constant = float(rate_law.equation.expression)
        except ValueError:
            raise ValueError('rate_law.equation.expression not a float')
        self.reactants = []
        self.reactant_coefficients = []
        for part in rate_law.reaction.participants:
            # only consider reactants
            if part.coefficient < 0:
                self.reactants.append(DynamicSpecies(dynamic_model, local_species_population,
                    part.species))
                self.reactant_coefficients.append(-part.coefficient)

    def calc_mass_action_rate(self, time, compartment_volume=None):
        """ Calculate a mass action rate

        Args:
            time (:obj:`float`): the current simulation time
            compartment_volume (:obj:`float`, optional): the volume of the compartment storing the
                reaction's reactants; optional volume
                enables amortization of the cost of computing volume over multiple rate calculations

        Returns:
            :obj:`float`: the rate for the rate law passed to `MassActionKinetics()` at time `time`
        """
        if self.order == 0:
            # zeroth order reaction
            combinations = 1
        elif self.order == 1:
            # uni reaction
            combinations = self.reactants[0].get_population(time)
        elif self.order == 2:
            specie_0_population = self.reactants[0].get_population(time)
            if self.reactant_coefficients[0] == 1:
                # since order == 2, both reactant_coefficients must be 1
                # bi- distinct reaction
                combinations = specie_0_population*self.reactants[1].get_population(time)
            elif self.reactant_coefficients[0] == 2:
                # bi- same reaction
                combinations = specie_0_population*(specie_0_population-1)/2
            else:
                assert False, \
                    "calc_mass_action_rate: reaction '{}': self.reactant_coefficients[0] ({}) not 1 or 2".format(
                        self.id, self.reactant_coefficients[0])
        else:
            assert False, "calc_mass_action_rate: reaction '{}': 2 < self.order ({})".format(self.id,
                self.order)

        if self.order == 1:
            rate = self.rate_constant * combinations
        else:
            if compartment_volume is None:
                compartment_volume = self.dynamic_compartment.volume()
            vol_avo = compartment_volume*Avogadro
            molarity_correction = pow(vol_avo, 1-self.order)
            rate = self.rate_constant * combinations * molarity_correction
        return rate

    @staticmethod
    def is_mass_action_rate_law(rate_law):
        """ Determine whether a rate law should use mass action kinetics

        Args:
            rate_law (:obj:`wc_lang.core.RateLaw`): a rate law

        Returns:
            :obj:`bool`: return `True` if `rate_law` should be evaluated using mass action
                kinetics, `False` otherwise
        """
        if hasattr(rate_law, 'k_cat') and not math.isnan(rate_law.k_cat):
            return False
        if hasattr(rate_law, 'k_m') and not math.isnan(rate_law.k_m):
            return False
        if not hasattr(rate_law, 'equation') or not hasattr(rate_law.equation, 'expression'):
            return False
        try:
            float(rate_law.equation.expression)
        except ValueError:
            return False
        molecularity = MassActionKinetics.molecularity(rate_law.reaction)
        if 2<molecularity:
            return False
        return True

    @staticmethod
    def molecularity(reaction):
        """ Determine the molecularity of a reaction

        Args:
            reaction (:obj:`wc_lang.core.Reaction`): a reaction

        Returns:
            :obj:`int`: the molecularity
        """
        molecularity = 0
        for part in reaction.participants:
            # only consider reactants
            if part.coefficient < 0:
                molecularity += -part.coefficient
        return molecularity

    def __str__(self):
        """ Provide a readable representation of this `MassActionKinetics`

        Returns:
            :obj:`str`: a readable representation of this `MassActionKinetics`
        """
        rv = ['MassActionKinetics:']
        rv.append("id: {}".format(self.id))
        rv.append("order: {}".format(self.order))
        rv.append("rate_constant: {}".format(self.rate_constant))
        rv.append("dynamic_compartment: {}".format(self.dynamic_compartment.id))
        rv.append("reactants: {}".format([s.id for s in self.reactants]))
        rv.append("reactant_coefficients: {}".format(self.reactant_coefficients))
        return '\n'.join(rv)


class DynamicRateLaw(DynamicComponent):
    """ The dynamic representation of a rate law

    Attributes:
        mass_action (:obj:`bool`): if set, use a mass action rate
        k_cat (:obj:`float`): if in `rate_law`, the law's k_cat, otherwise not set
        k_m (:obj:`float`): if in `rate_law`, the law's k_m, otherwise not set
        transcoded_equation (:obj:`float`): if not a mass action rate law, the rate law's
            equation transcoded for evaluation
    """

    def __init__(self, dynamic_model, local_species_population, dynamic_compartment, rate_law):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            dynamic_compartment (:obj:`DynamicCompartment`): the `DynamicCompartment` that contains
                the reactants in and modifiers for the reaction
            rate_law (:obj:`wc_lang.core.RateLaw`): the corresponding `wc_lang` `RateLaw`
        """
        super().__init__(dynamic_model, local_species_population, rate_law)
        self.dynamic_compartment = dynamic_compartment
        if MassActionKinetics.is_mass_action_rate_law(rate_law):
            self.mass_action = True
            self.mass_action_rate_law = MassActionKinetics(dynamic_model, local_species_population,
                dynamic_compartment, rate_law)
        else:
            self.mass_action = False
            if hasattr(rate_law, 'k_cat') and not math.isnan(rate_law.k_cat):
                self.k_cat = rate_law.k_cat
            if hasattr(rate_law, 'k_m') and not math.isnan(rate_law.k_m):
                self.k_m = rate_law.k_m
            self.transcoded_equation = RateLawUtils.transcode(rate_law.equation,
                dynamic_model.dynamic_species.keys(), dynamic_model.dynamic_parameters.keys())

    def eval(self, time, parameter_values=None, species_concentrations=None, compartment_volume=None):
        """ Provide this rate law's rate

        Args:
            time (:obj:`float`): the current simulation time
            parameter_values (:obj:`dict`, optional): a map: parameter_id -> parameter value
            compartment_volume (:obj:`float`, optional): the volume of the compartment storing the
                reaction's reactants; optional volume; must be provided if `self.mass_action` is False
                enables amortization of the cost of computing volume over multiple rate calculations
            species_concentrations (:obj:`dict`, optional): species_id -> species concentration; must
                be provided if `self.mass_action` is False

        Returns:
            :obj:`float`: the rate for the rate law at time `time`
        """
        if self.mass_action:
            return self.mass_action_rate_law.calc_mass_action_rate(time, compartment_volume)
        else:
            return RateLawUtils.eval_rate_law(self, species_concentrations, parameter_values,
                transcoded_equation=self.transcoded_equation)

    def __str__(self):
        """ Provide a readable representation of this `DynamicRateLaw`

        Returns:
            :obj:`str`: a readable representation of this `DynamicRateLaw`
        """
        rv = ['DynamicRateLaw:']
        rv.append("id: {}".format(self.id))
        rv.append("dynamic_compartment: {}".format(self.dynamic_compartment.id))
        rv.append("mass_action: {}".format(self.mass_action))
        for attr in ['k_m', 'k_cat', 'transcoded_equation']:
            if hasattr(self, attr):
                rv.append("{}: {}".format(attr, getattr(self, attr)))
        return '\n'.join(rv)


class DynamicReaction(DynamicComponent):
    """ A dynamic reaction

    A `DynamicReaction` represents a single reaction in the simulator.

    Attributes:
        dynamic_rate_law (:obj:`DynamicRateLaw`): the reaction's rate law
    """
    def __init__(self, dynamic_model, local_species_population, reaction):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            reaction (:obj:`wc_lang.core.Reaction`): the corresponding `wc_lang` `Reaction`

        Raises:
            :obj:`MultialgorithmError`: if the modifiers in `reaction`'s rate law are contained in
                multiple compartments
        """
        super().__init__(dynamic_model, local_species_population, reaction)

        # prepare this reaction's rate law
        compartment = self.verify_reactants_and_modifiers_compartment(reaction)
        dynamic_compartment = dynamic_model.dynamic_compartments[compartment.id]
        self.dynamic_rate_law = DynamicRateLaw(dynamic_model, local_species_population, dynamic_compartment,
            reaction.rate_laws[0])

    @staticmethod
    def verify_reactants_and_modifiers_compartment(reaction):
        """ Verify the compartment used by a reaction's reactants and rate law modifiers

        Args:
            reaction (:obj:`wc_lang.core.Reaction`): a `wc_lang` `Reaction`

        Returns:
            :obj:`wc_lang.core.Compartment`: returns the compartment containing `reaction`'s
                reactants and rate law modifiers

        Raises:
            :obj:`MultialgorithmError`: if `reaction`'s reactants and rate law modifiers are contained
                in multiple compartments, or if they are contained in a compartment other than
                the reaction's submodel's compartment
        """
        compartments = set()
        if hasattr(reaction.rate_laws[0].equation, 'modifiers'):
            for modifier in reaction.rate_laws[0].equation.modifiers:
                compartments.add(modifier.compartment)
        for participant in reaction.participants:
            compartments.add(participant.species.compartment)
        if 1<len(compartments):
            raise MultialgorithmError("reactants and modifiers of reaction '{}' reside in multiple compartments: {}".format(
                reaction.id, ', '.join([comp.id for comp in compartments])))
        elif 1 == len(compartments):
            compartment = compartments.pop()
            # check that the compartment is the compartment for reaction.submodel
            if compartment != reaction.submodel.compartment:
                raise MultialgorithmError("reactants and/or modifiers of reaction '{}' are stored in "
                    "'{}', which is not its submodel's compartment: '{}'".format(
                    reaction.id, compartment.id, reaction.submodel.compartment.id))
            return reaction.submodel.compartment
        else:
            return reaction.submodel.compartment

    def __str__(self):
        """ Provide a readable representation of this `DynamicReaction`

        Returns:
            :obj:`str`: a readable representation of this `DynamicReaction`
        """
        rv = ['DynamicRateLaw:']
        rv.append("id: {}".format(self.id))
        rv.append("dynamic_rate_law.id: {}".format(self.dynamic_rate_law.id))
        return '\n'.join(rv)


WC_LANG_MODEL_TO_DYNAMIC_MODEL = {
    Function: DynamicFunction,
    Parameter: DynamicParameter,
    Species: DynamicSpecies,
    Observable: DynamicObservable,
    StopCondition: DynamicStopCondition,
    ObjectiveFunction: DynamicObjectiveFunction,
    Reaction: DynamicReaction,
    RateLaw: DynamicRateLaw
}
