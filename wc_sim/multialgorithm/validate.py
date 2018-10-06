"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-09-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import re
import tempfile
import shutil
import warnings
from enum import Enum
import pandas as pd
import numpy as np
import inspect
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from wc_lang.io import Reader
from wc_lang.core import ReactionParticipantAttribute
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults
from wc_sim.multialgorithm.config import core as config_core_multialgorithm
config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


# todo: doc strings
class Error(Exception):
    """ Base class for exceptions involving `wc_sim` validation

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class ValidationError(Error):
    """ Exception raised for errors in `wc_sim.multialgorithm.validate`

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class WcSimValidationWarning(UserWarning):
    """ `wc_sim` Validation warning """
    pass


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

TEST_CASE_TYPE_TO_DIR = {
    'continuous_deterministic': 'semantic',
    'discrete_stochastic': 'stochastic'}


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
        # convert all numerics to floats
        for key, value in settings.items():
            try:
                settings[key] = float(value)
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
    TOLERANCE_MAP = dict(
        rtol='relative',
        atol='absolute'
    )

    def __init__(self, validation_test_reader, simulation_run_results):
        self.validation_test_reader = validation_test_reader
        self.strip_compartments(simulation_run_results)
        self.simulation_run_results = simulation_run_results
        # obtain default tolerances in np.allclose()
        self.default_tolerances = ValidationUtilities.get_default_args(np.allclose)

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
        for np_name, testing_name in self.TOLERANCE_MAP.items():
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
            # todo: warn if values lack precision; want int64 integers and float64 floats
            # see https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.scalars.html
            # use warnings.warn("", WcSimValidationWarning)

            ### test means ###
            mean_range = self.validation_test_reader.settings['meanRange']
            n_runs = len(self.simulation_run_results)

            times = self.simulation_run_results[0].get('populations').index
            n_times = len(times)

            self.simulation_pop_means = {}
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

                # load simul. runs into 2D np array to find mean
                run_results_array = np.empty((n_times, n_runs))
                for idx, run_result in enumerate(self.simulation_run_results):
                    run_pop_df = run_result.get('populations')
                    run_results_array[:, idx] = run_pop_df.loc[:, species_type].values
                # average simul. run populations over times
                self.simulation_pop_means[species_type] = pop_mean = run_results_array.mean(axis=1)
                Z = math.sqrt(n_runs) * (pop_mean - e_mean) / e_sd

                # compare with mean_range
                if np.any(Z < mean_range[0]) or np.any(mean_range[1] < Z):
                    differing_values.append(species_type)

            ### test sds ###
            # todo: test sds
            return differing_values or False


class CaseValidator(object):
    """ Validate a test case """
    def __init__(self, test_cases_root_dir, test_case_type, test_case_num,
        default_num_stochastic_runs=config_multialgorithm['num_ssa_validation_sim_runs']):
        # read model, config and expected predictions
        self.test_case_dir = os.path.join(test_cases_root_dir, TEST_CASE_TYPE_TO_DIR[test_case_type],
            test_case_num)
        self.validation_test_reader = ValidationTestReader(test_case_type, self.test_case_dir, test_case_num)
        self.validation_test_reader.run()
        self.default_num_stochastic_runs = default_num_stochastic_runs

    def validate_model(self, num_discrete_stochastic_runs=None, discard_run_results=True, plot_file=None):
        """ Validate a model
        """
        # todo: make this work for continuous_deterministic models
        '''
            todo: retry on failure
                if failure, retry "evaluate whether mean of simulation trajectories match expected trajectory"
                    # simulations generating the correct trajectories will fail validation (100*(p-value threshold)) percent of the time
                if failure again, report failure    # assuming p-value << 1, two failures indicates likely errors
        '''
        # todo: convert to probabilistic test with multiple runs and p-value
        ## 1. run simulation
        # check settings
        required_settings = ['duration', 'steps']
        settings = self.validation_test_reader.settings
        errors = []
        for setting in required_settings:
            if setting not in settings:
                errors.append("required setting '{}' not provided".format(setting))
            elif not isinstance(settings[setting], float):
                errors.append("required setting '{}' not a float".format(setting))
        if errors:
            raise ValidationError('; '.join(errors))
        if 'start' in settings and settings['start'] != 0:
            raise ValidationError("non-zero start setting ({}) not supported".format(settings['start']))

        # run simulation
        self.tmp_results_dir = tmp_results_dir = tempfile.mkdtemp()
        simul_kwargs = dict(end_time=settings['duration'],
            checkpoint_period=settings['duration']/settings['steps'],
            results_dir=tmp_results_dir)

        if self.validation_test_reader.test_case_type == ValidationTestCaseType.continuous_deterministic:
            simulation = Simulation(self.validation_test_reader.model)
            _, results_dir = simulation.run(**simul_kwargs)
            simulation_run_results = RunResults(results_dir)

        if self.validation_test_reader.test_case_type == ValidationTestCaseType.discrete_stochastic:
            # make multiple simulation runs with different random seeds
            if num_discrete_stochastic_runs is not None:
                num_runs = num_discrete_stochastic_runs
            else:
                num_runs = self.default_num_stochastic_runs
            self.num_runs = num_runs
            self.simulation_run_results = []
            for _ in range(num_runs):
                simul_kwargs['results_dir'] = tempfile.mkdtemp(dir=tmp_results_dir)
                # todo: provide method(s) in Simulation and classes it uses (SimulationEngine) to reload() a simulation,
                # that is, do another monte Carlo simulation with a different seed
                simulation = Simulation(self.validation_test_reader.model)
                _, results_dir = simulation.run(**simul_kwargs)
                self.simulation_run_results.append(RunResults(results_dir))

        ## 2. compare results
        self.results_comparator = ResultsComparator(self.validation_test_reader, self.simulation_run_results)
        self.comparison_result = self.results_comparator.differs()

        ## 3 plot comparison of actual and expected trajectories
        if plot_file:
            self.plot_model_validation(plot_file)
        # todo: optionally, save results
        # todo: output difference between actual and expected trajectory

        ## 4. cleanup
        if discard_run_results:
            shutil.rmtree(self.tmp_results_dir)
        return self.comparison_result

    def get_model_summary(self):
        """Summarize the test model
        """
        mdl = self.validation_test_reader.model
        summary = ['Model Summary:']
        summary.append("model {}: {}".format(mdl.id, mdl.name))
        for cmpt in mdl.compartments:
            summary.append("compartment {}: {}, init. vol. {}".format(cmpt.id, cmpt.name,
                cmpt.initial_volume))
        reaction_participant_attribute = ReactionParticipantAttribute()
        for sm in mdl.submodels:
            summary.append("submodel {}: in {}".format(sm.id, sm.compartment.id))
            for rxn in sm.reactions:
                summary.append("reaction and rate law {}: {}, {}".format(rxn.id,
                    reaction_participant_attribute.serialize(rxn.participants),
                    rxn.rate_laws[0].equation.serialize()))
        for param in mdl.get_parameters():
            summary.append("param: {}={} ({})".format(param.id, param.value, param.units))
        return summary

    def get_test_case_summary(self):
        """Summarize the test case
        """
        summary = ['Test Case Summary']
        if self.comparison_result:
            summary.append("Failing species types: {}".format(', '.join(self.comparison_result)))
        else:
            summary.append('All species types validate')
        summary.append("Num simul runs: {}".format(self.num_runs))
        summary.append("Test case type: {}".format(self.validation_test_reader.test_case_type.name))
        summary.append("Test case number: {}".format(self.validation_test_reader.test_case_num))
        return summary

    def plot_model_validation(self, plot_file):
        """Plot a model validation run
        """
        # todo: make this work for continuous_deterministic models
        # todo: use matplotlib 3; use the more flexible OO API instead of pyplot
        # todo: optional for actual pops:(may be too many); alternatively a random pct of actuals, or the density of actuals
        times = self.simulation_run_results[0].get('populations').index
        for species_type in self.validation_test_reader.settings['amount']:

            # plot mean simulation pop
            model_mean, = plt.plot(times, self.results_comparator.simulation_pop_means[species_type], 'g-')

            # plot simulation pops
            for rr in self.simulation_run_results:
                pop_time_series = rr.get('populations').loc[:, species_type]
                simul_pops, = plt.plot(times, pop_time_series, 'b-', linewidth=0.1)

            # plot expected predictions
            expected_mean_df = self.validation_test_reader.expected_predictions_df.loc[:, species_type+'-mean']
            correct_mean, = plt.plot(times, expected_mean_df.values, 'r-')
            # mean +/- 3 sd
            expected_sd_df = self.validation_test_reader.expected_predictions_df.loc[:, species_type+'-sd']
            kwargs = dict(linewidth=0.4, color='brown')
            correct_mean_plus_3sd, = plt.plot(times, expected_mean_df.values + 3 * expected_sd_df, **kwargs)
            correct_mean_minus_3sd, = plt.plot(times, expected_mean_df.values - 3 * expected_sd_df, **kwargs)

            plt.ylabel('population')
            plt.xlabel('time (s)')
            plt.legend((simul_pops, model_mean, correct_mean, correct_mean_plus_3sd),
                ('{} simul runs'.format(species_type), '{} simul mean'.format(species_type), 'correct mean',
                    '3 sd from correct mean'),
                loc='lower left', fontsize=10)
            axes = plt.gca()
            summary = self.get_model_summary()
            middle = len(summary)//2
            x_pos = 0
            y_pos = 1.02
            for lb, ub in [(0, middle), (middle, len(summary))]:
                text = plt.text(x_pos, y_pos, '\n'.join(summary[lb:ub]), fontsize=4, transform=axes.transAxes)
                # todo: position text automatically
                x_pos += 0.4
            test_case_summary = self.get_test_case_summary()
            plt.text(0.8, y_pos, '\n'.join(test_case_summary), fontsize=4, transform=axes.transAxes)
            fig = plt.gcf()
            fig.savefig(plot_file)
            plt.close(fig)
            return "Wrote: {}".format(plot_file)


class ValidationSuite(object):
    """ Run suite of validation tests of `wc_sim`'s dynamic behavior """
    pass


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
