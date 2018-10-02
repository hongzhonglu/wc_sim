"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-09-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import re
import warnings
from enum import Enum
import pandas as pd
import numpy as np
import inspect
import math

from wc_lang.io import Reader
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.run_results import RunResults


# todo: doc strings
class Error(Exception):
    """ Base class for exceptions involving `wc_sim` validation

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class ValidationError(Error):
    """ Exception raised for errors in `wc_sim.multialgorithm.validate

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class SubmodelValidator(object):
    """ Validate dynamic behavior of a single `wc_sim` submodel """
    pass


class SsaValidator(SubmodelValidator):
    """ Validate dynamic behavior of the `wc_sim` SSA submodel """
    pass


class FbaValidator(SubmodelValidator):
    """ Validate dynamic behavior of the `wc_sim` FBA submodel """
    pass


class OdeValidator(SubmodelValidator):
    """ Validate dynamic behavior of the `wc_sim` ODE submodel """
    pass


class ValidationTestCaseType(Enum):
    """ Types of test cases """
    continuous_deterministic = 1    # algorithms like ODE
    discrete_stochastic = 2         # algorithms like SSA


class ValidationTestReader(object):
    """ Read a model validation test case """
    SBML_FILE_SUFFIX = '.xml'
    def __init__(self, test_case_type, test_case_dir, test_case_num):
        if test_case_type not in ValidationTestCaseType.__members__:
            raise ValidationError("Unknown ValidationTestCaseType: '{}'".format(test_case_type))
        else:
            self.test_case_type = ValidationTestCaseType[test_case_type]
        self.test_case_dir = test_case_dir
        self.test_case_num = test_case_num

    def read_settings(self):
        """ Read a test case's settings into a key-value dictionary """
        self.settings_file = settings_file = os.path.join(self.test_case_dir, self.test_case_num+'-settings.txt')
        settings = {}
        errors = []
        try:
            with open(settings_file, 'r') as f:
                for line in f:
                    if line.strip():
                        key, value = line.strip().split(':', maxsplit=1)
                        if key in settings:
                            errors.append("duplicate key '{}' in settings file '{}'".format(key, settings_file))
                        settings[key] = value.strip()
        except Exception as e:
            errors.append("could not read settings file '{}': {}".format(settings_file, e))
        if errors:
            raise ValidationError('; '.join(errors))

        # convert settings values
        # convert numerics to floats
        for key, value in settings.items():
            try:
                f = float(value)
                settings[key] = f
            except:
                pass
        # split into lists
        for key in ['variables', 'amount', 'concentration']:
            if key in settings and settings[key]:
                settings[key] = re.split(r'\W+', settings[key])
        # convert meanRange and sdRange into numeric tuples
        for key in ['meanRange', 'sdRange']:
            if key in settings and settings[key]:
                settings[key] = eval(settings[key])
        return settings

    def read_expected_predictions(self):
        self.expected_predictions_file = expected_predictions_file = os.path.join(
            self.test_case_dir, self.test_case_num+'-results.csv')
        expected_predictions_df = pd.read_csv(expected_predictions_file)
        # expected predictions should contain data for all time steps
        times = np.linspace(self.settings['start'], self.settings['duration'], num=self.settings['steps']+1)
        if not np.allclose(times, expected_predictions_df.time):
            raise ValidationError("times in settings '{}' differ from times in expected predictions '{}'".format(
                self.settings_file, expected_predictions_file))

        if self.test_case_type == ValidationTestCaseType.continuous_deterministic:
            # expected predictions should contain the mean and sd of each variable in 'amount'
            expected_columns = set(self.settings['amount'])
            actual_columns = set(expected_predictions_df.columns.values)
            if expected_columns - actual_columns:
                raise ValidationError("some amounts missing from expected predictions '{}': {}".format(
                    expected_predictions_file, expected_columns - actual_columns))

        if self.test_case_type == ValidationTestCaseType.discrete_stochastic:
            # expected predictions should contain the mean and sd of each variable in 'amount'
            expected_columns = set()
            for amount in self.settings['amount']:
                expected_columns.add(amount+'-mean')
                expected_columns.add(amount+'-sd')
            if expected_columns - set(expected_predictions_df.columns.values):
                raise ValidationError("mean or sd of some amounts missing from expected predictions '{}': {}".format(
                    expected_predictions_file, expected_columns - set(expected_predictions_df.columns.values)))

        return expected_predictions_df

    def read_model(self):
        """  Read a model into a `wc_lang` representation. """
        self.model_filename = model_filename = os.path.join(
            self.test_case_dir, self.test_case_num+'-wc_lang.xlsx')
        if model_filename.endswith(self.SBML_FILE_SUFFIX):   # pragma: no cover
            raise ValidationError("Reading SBML files not supported: model filename '{}'".format(model_filename))
        return Reader().run(self.model_filename, strict=False)

    def run(self):
        self.settings = self.read_settings()
        self.expected_predictions_df = self.read_expected_predictions()
        self.model = self.read_model()


# the compartment for test cases
TEST_CASE_COMPARTMENT = 'c'

class ResultsComparator(object):
    """ Compare simulated and expected predictions """
    def __init__(self, validation_test_reader, simulation_run_results):
        self.validation_test_reader = validation_test_reader
        self.strip_compartments(simulation_run_results)
        self.simulation_run_results = simulation_run_results
        # obtain default tolerances
        self.default_tolerances = ValidationUtilities.get_default_args(np.allclose)
        self.tolerance_map = dict(
            rtol='relative',
            atol='absolute'
        )

    @staticmethod
    def get_species(species_type):
        return "{}[{}]".format(species_type, TEST_CASE_COMPARTMENT)

    @staticmethod
    def get_species_type(string):
        # if string is a species get the species type, otherwise return the string
        if re.match(r'\w+\[\w+\]$', string, flags=re.ASCII):
            return string.split('[')[0]
        return string

    @classmethod
    def strip_compartments(cls, simulation_run_results):
        # if they're present, strip compartments from simulation run results; assumes all validation tests happen in one compartment
        # note that original run results are changed; assumes that is OK because they're used just for validation
        if isinstance(simulation_run_results, RunResults):
            populations_df = simulation_run_results.get('populations')
            col_names = list(populations_df.columns)
            populations_df.columns = list(map(cls.get_species_type, col_names))
        elif isinstance(simulation_run_results, list):
            for run_result in simulation_run_results:
                cls.strip_compartments(run_result)
        else:
            raise ValidationError("wrong type for simulation_run_results '{}'".format(
                type(simulation_run_results)))

    def prepare_tolerances(self):
        """ Prepare tolerance dictionary

        Use values from `validation_test_reader.settings` if available, otherwise from `numpy.allclose()`s defaults

        Returns:
            :obj:`dict`: kwargs for `rtol` and `atol` tolerances for use by `numpy.allclose()`
        """
        kwargs = {}
        for np_name, testing_name in self.tolerance_map.items():
            kwargs[np_name] = self.default_tolerances[np_name]
            if testing_name in self.validation_test_reader.settings:
                try:
                    tolerance = float(self.validation_test_reader.settings[testing_name])
                    kwargs[np_name] = tolerance
                except:
                    pass
        return kwargs

    @staticmethod
    def zero_to_inf(np_array):
        """ replace 0s with inf """
        infs = np.full(np_array.shape, float('inf'))
        return np.where(np_array != 0, np_array, infs)

    # todo: check concentrations too
    def differs(self):
        """ Evaluate whether simulation runs(s) differ from their expected species population prediction(s)

        Returns:
            :obj:`obj`: `False` if populations in the expected result and simulation run are equal
                within tolerances, otherwise :obj:`list`: of species with differing values
        """
        differing_values = []
        if self.validation_test_reader.test_case_type == ValidationTestCaseType.continuous_deterministic:
            kwargs = self.prepare_tolerances()
            populations_df = self.simulation_run_results.get('populations')
            # for each prediction, determine whether its trajectory is close enough to the expected predictions
            for species_type in self.validation_test_reader.settings['amount']:
                if not np.allclose(self.validation_test_reader.expected_predictions_df[species_type],
                    populations_df[species_type], **kwargs):
                    differing_values.append(species_type)
            return differing_values or False

        if self.validation_test_reader.test_case_type == ValidationTestCaseType.discrete_stochastic:
            """ Test mean and sd population over multiple runs

            Follow algorithm in
            github.com/sbmlteam/sbml-test-suite/blob/master/cases/stochastic/DSMTS-userguide-31v2.pdf,
            from Evans, et al. The SBML discrete stochastic models test suite, Bioinformatics, 24:285-286, 2008.
            """
            # todo: warn if dtypes are not int64
            range = self.validation_test_reader.settings['meanRange']
            n_runs = len(self.simulation_run_results)

            # todo: put this in RunResults method
            n_times = len(self.simulation_run_results[0].get('populations').index)

            for species_type in self.validation_test_reader.settings['amount']:
                # extract nx1 correct mean and sd np arrays
                correct_df = self.validation_test_reader.expected_predictions_df
                e_mean = correct_df.loc[:, species_type+'-mean'].values
                e_sd = correct_df.loc[:, species_type+'-sd'].values
                # errors if e_sd or e_mean < 0
                if np.any(e_mean < 0):
                    raise ValidationError("e_mean contains negative value(s)")
                if np.any(e_sd < 0):
                    raise ValidationError("e_sd contains negative value(s)")

                # avoid division by 0 and sd=0; replace 0s in e_sd with inf
                e_sd = self.zero_to_inf(e_sd)

                # load simul. runs into 2D np array
                run_results_array = np.empty((n_times, n_runs))
                for idx, run_result in enumerate(self.simulation_run_results):
                    run_pop_df = run_result.get('populations')
                    run_results_array[:, idx] = run_pop_df.loc[:, species_type].values

                # average simul. run populations over times
                pop_mean = run_results_array.mean(axis=1)
                Z = math.sqrt(n_runs) *(pop_mean - e_mean)/e_sd

                # compare with range
                if np.any(Z<range[0]) or np.any(range[1]<Z):
                    differing_values.append(species_type)
            return differing_values or False


class ValidationTestRunner(object):
    """ Run one validation case """
    '''
        1. read model, config and expected predictions
        2. run simulation
        3. compare results
        4. save results
        outputs:
            submodel success or failure
            measure of difference between actual and expected trajectory
            plotted comparison of actual and expected trajectories
    '''
    '''
    test single model
        inputs: test submodel, submodel determinism, expected trajectory, [# simulations, p-value threshold]
            if failure, retry "evaluate whether mean of simulation trajectories match expected trajectory"
                # simulations generating the correct trajectories will fail validation (100*(p-value threshold)) percent of the time
            if failure again, report failure    # assuming p-value << 1, two failures indicates likely errors
    '''
    def __init__(self, test_case_type, test_case_dir, test_case_num):
        pass


class ValidationSuite(object):
    """ Run suite of validation tests of `wc_sim`'s dynamic behavior """
    pass


# todo: unittest get_default_args
class ValidationUtilities(object):

    @staticmethod
    def get_default_args(func):
        """ Get the names and default values of function's keyword arguments

        From https://stackoverflow.com/a/12627202

        Args:
            func (:obj:`Function`): a Python function

        Returns:
            :obj:`dict`: a map: name -> default value
        """
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
