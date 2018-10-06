"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-09-25
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import tempfile
import os
import shutil
import pandas
import numpy as np
import math
import datetime

import obj_model
from wc_sim.multialgorithm.validate import (ValidationError, SubmodelValidator, SsaValidator, FbaValidator,
    OdeValidator, ValidationTestCaseType, ValidationTestReader, ResultsComparator,
    ResultsComparator, CaseValidator, ValidationSuite, ValidationUtilities, TEST_CASE_TYPE_TO_DIR,
    TEST_CASE_COMPARTMENT)
from wc_sim.multialgorithm.run_results import RunResults

TEST_CASES = os.path.join(os.path.dirname(__file__), 'fixtures', 'validation', 'test_cases')

def make_test_case_dir(test_case_num, test_case_type='discrete_stochastic'):
    return os.path.join(TEST_CASES, TEST_CASE_TYPE_TO_DIR[test_case_type], test_case_num)

def make_validation_test_reader(test_case_num, test_case_type='discrete_stochastic'):
    return ValidationTestReader(test_case_type, make_test_case_dir(test_case_num, test_case_type), test_case_num)


class TestValidationTestReader(unittest.TestCase):

    def test_read_settings(self):
        settings = make_validation_test_reader('00001').read_settings()
        some_expected_settings = dict(
            start=0,
            variables=['X'],
            amount=['X'],
            sdRange=(-5, 5)
        )
        for expected_key, expected_value in some_expected_settings.items():
            self.assertEqual(settings[expected_key], expected_value)
        settings = make_validation_test_reader('00003').read_settings()
        self.assertEqual(settings['key1'], 'value1: has colon')
        self.assertEqual(
            make_validation_test_reader('00004').read_settings()['amount'], ['X', 'Y'])

        with self.assertRaisesRegexp(ValidationError,
            "duplicate key 'key1' in settings file '.*00002/00002-settings.txt"):
            make_validation_test_reader('00002').read_settings()

        with self.assertRaisesRegexp(ValidationError,
            "could not read settings file.*00005/00005-settings.txt.*No such file or directory.*"):
            make_validation_test_reader('00005').read_settings()

    def test_read_expected_predictions(self):
        for test_case_type in ['discrete_stochastic', 'continuous_deterministic']:
            validation_test_reader = make_validation_test_reader('00001', test_case_type=test_case_type)
            validation_test_reader.settings = validation_test_reader.read_settings()
            expected_predictions_df = validation_test_reader.read_expected_predictions()
            self.assertTrue(isinstance(expected_predictions_df, pandas.core.frame.DataFrame))

        # wrong time sequence
        validation_test_reader.settings['duration'] += 1
        with self.assertRaisesRegexp(ValidationError, "times in settings .* differ from times in expected predictions"):
            validation_test_reader.read_expected_predictions()
        validation_test_reader.settings['duration'] -= 1

        # wrong columns
        missing_variable = 'MissingVariable'
        for test_case_type, expected_error in [
            ('discrete_stochastic', "mean or sd of some amounts missing from expected predictions.*{}"),
            ('continuous_deterministic', "some amounts missing from expected predictions.*{}")]:
            validation_test_reader = make_validation_test_reader('00001', test_case_type=test_case_type)
            validation_test_reader.settings = validation_test_reader.read_settings()
            validation_test_reader.settings['amount'].append(missing_variable)
            with self.assertRaisesRegexp(ValidationError, expected_error.format(missing_variable)):
                validation_test_reader.read_expected_predictions()

    def test_read_model(self):
        validation_test_reader = make_validation_test_reader('00001')
        model = validation_test_reader.read_model()
        self.assertTrue(isinstance(model, obj_model.Model))
        self.assertEqual(model.id, 'test_case_' + validation_test_reader.test_case_num)

    def test_validation_test_reader(self):
        test_case_num = '00001'
        validation_test_reader = make_validation_test_reader(test_case_num)
        self.assertEqual(None, validation_test_reader.run())

        # exceptions
        with self.assertRaisesRegexp(ValidationError, "Unknown ValidationTestCaseType:"):
            ValidationTestReader('no_such_test_case_type', '', test_case_num)


class TestResultsComparator(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.test_case_type = 'continuous_deterministic'
        self.test_case_num = '00001'
        self.simulation_run_results = self.make_run_results_from_expected_results(self.test_case_type,
            self.test_case_num)

    def make_run_results_filename(self):
        return os.path.join(tempfile.mkdtemp(dir=self.tmp_dir), 'run_results.h5')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def make_run_results_from_expected_results(self, test_case_type, test_case_num):
        """ Create a RunResults object with the same population as a test case """
        results_file = os.path.join(TEST_CASES, TEST_CASE_TYPE_TO_DIR[test_case_type], test_case_num,
            test_case_num+'-results.csv')
        results_df = pandas.read_csv(results_file)
        return self.make_run_results(results_df, add_compartments=True)

    def test_results_comparator_continuous_deterministic(self):
        validation_test_reader = make_validation_test_reader(self.test_case_num, self.test_case_type)
        validation_test_reader.run()
        results_comparator = ResultsComparator(validation_test_reader, self.simulation_run_results)
        self.assertEqual(False, results_comparator.differs())

        # modify the run results for first specie at time 0
        amount_1 = validation_test_reader.settings['amount'][0]
        self.simulation_run_results.get('populations').loc[0, amount_1] += 1
        self.assertTrue(results_comparator.differs())
        self.assertIn(amount_1, results_comparator.differs())

        '''
        # test functionality of tolerances
        # allclose condition: absolute(a - b) <= (atol + rtol * absolute(b))
        # b is populations
        # assume b < a and 0 < b
        # condition becomes, a - b <= atol + rtol * b
        # or, a <= atol + (1 + rtol) * b        (1)
        # or, (a - atol)/(1 + rtol) <= b        (2)
        # use (1) or (2) to pick a given b, or b given a, respectively
        # modify b:
        # b = (a - atol)/(1 + rtol)     ==> equals
        # b += epsilon                  ==> differs
        '''

    def make_run_results(self, pops_df, add_compartments=False):
        # make a RunResults obj with given population
        if add_compartments:
            # add compartments to species
            cols = list(pops_df.columns)
            pops_df.columns = cols[:1] + list(map(ResultsComparator.get_species, cols[1:]))

        # create an hdf
        run_results_filename = self.make_run_results_filename()
        pops_df.to_hdf(run_results_filename, 'populations')

        # add the other RunResults components as empty dfs
        empty_df = pandas.DataFrame(index=[], columns=[])
        for component in RunResults.COMPONENTS:
            if component != 'populations':
                empty_df.to_hdf(run_results_filename, component)

        # make & return a RunResults
        return RunResults(os.path.dirname(run_results_filename))

    def test_strip_compartments(self):
        pop_df = pandas.DataFrame(np.ones((2, 3)), index=[0, 10], columns='time X Y'.split())
        run_results = self.make_run_results(pop_df, add_compartments=True)
        pop_columns = list(run_results.get('populations').columns[1:])
        self.assertTrue(all(['[' in s for s in pop_columns]))
        ResultsComparator.strip_compartments(run_results)
        pop_columns = list(run_results.get('populations').columns[1:])
        self.assertFalse(all(['[' in s for s in pop_columns]))
        ResultsComparator.strip_compartments(run_results)
        pop_columns = list(run_results.get('populations').columns[1:])
        self.assertFalse(all(['[' in s for s in pop_columns]))

        rrs = []
        for i in range(2):
            rrs.append(self.make_run_results(pop_df))
        ResultsComparator.strip_compartments(rrs)
        for rr in rrs:
            pop_columns = list(rr.get('populations').columns[1:])
            self.assertFalse(all(['[' in s for s in pop_columns]))

        with self.assertRaisesRegexp(ValidationError, "wrong type for simulation_run_results.*"):
            ResultsComparator.strip_compartments(1.0)

    def stash_pd_value(self, df, loc, new_val):
        self.stashed_pd_value = df.loc[loc[0], loc[1]]
        df.loc[loc[0], loc[1]] = new_val

    def restore_pd_value(self, df, loc):
        df.loc[loc[0], loc[1]] = self.stashed_pd_value

    def test_results_comparator_discrete_stochastic(self):
        # todo: move code that's common with test_results_comparator_continuous_deterministic to function
        # todo: test multiple amount variables
        # todo: consolidate setup
        test_case_type = 'discrete_stochastic'
        test_case_num = '00001'
        validation_test_reader = make_validation_test_reader(test_case_num, test_case_type)
        validation_test_reader.run()

        # make multiple run_results with variable populations
        # todo: use make_run_results_from_expected_results()
        # simulation_run_results = self.make_run_results_from_expected_results(test_case_type, test_case_num)
        expected_predictions_df = validation_test_reader.expected_predictions_df
        times = expected_predictions_df.loc[:,'time'].values
        n_times = len(times)
        means = expected_predictions_df.loc[:,'X-mean'].values
        correct_pop = expected_predictions_df.loc[:,['time', 'X-mean']]
        n_runs = 3
        run_results = []
        for i in range(n_runs):
            pops = np.empty((n_times, 2))
            pops[:,0] = times
            # add stochasticity to populations
            pops[:,1] = np.add(means, (np.random.rand(n_times)*2)-1)
            pop_df = pandas.DataFrame(pops, index=times, columns=['time', 'X'])
            run_results.append(self.make_run_results(pop_df, add_compartments=True))
            new_run_results_pop = run_results[-1].get('populations')

        results_comparator = ResultsComparator(validation_test_reader, run_results)
        self.assertEqual(False, results_comparator.differs())

        ### adjust data to test all Z thresholds ###
        # choose an arbitrary time
        time = 10

        # Z                 differ?
        # -                 -------
        # range[0]-epsilon  yes
        # range[0]+epsilon  no
        # range[1]-epsilon  no
        # range[1]+epsilon  yes
        epsilon = 1E-9
        lower_range, upper_range = (validation_test_reader.settings['meanRange'][0],
            validation_test_reader.settings['meanRange'][1])
        z_scores_and_expected_differs = [
            (lower_range - epsilon, ['X']),
            (lower_range + epsilon, False),
            (upper_range - epsilon, False),
            (upper_range + epsilon, ['X'])
        ]

        def get_test_pop_mean(time, n_runs, expected_df, desired_Z):
            # solve Z = math.sqrt(n_runs)*(pop_mean - e_mean)/e_sd for pop_mean:
            # pop_mean = Z*e_sd/math.sqrt(n_runs) + e_mean
            # return pop_mean for desired_Z
            return desired_Z * expected_df.loc[time, 'X-sd'] / math.sqrt(n_runs) + \
                expected_df.loc[time, 'X-mean']

        def set_all_pops(run_results_list, time, pop_val):
            # set all pops to pop_val
            for rr in run_results_list:
                rr.get('populations').loc[time, 'X'] = pop_val

        for test_z_score, expected_differ in z_scores_and_expected_differs:
            test_pop_mean = get_test_pop_mean(time, n_runs, expected_predictions_df, test_z_score)
            set_all_pops(run_results, time, test_pop_mean)
            results_comparator = ResultsComparator(validation_test_reader, run_results)
            self.assertEqual(expected_differ, results_comparator.differs())

        loc = [0, 'X-mean']
        self.stash_pd_value(expected_predictions_df, loc, -1)
        with self.assertRaisesRegexp(ValidationError, "e_mean contains negative value.*"):
            ResultsComparator(validation_test_reader, run_results).differs()
        self.restore_pd_value(expected_predictions_df, loc)

        loc = [0, 'X-sd']
        self.stash_pd_value(expected_predictions_df, loc, -1)
        with self.assertRaisesRegexp(ValidationError, "e_sd contains negative value.*"):
            ResultsComparator(validation_test_reader, run_results).differs()
        self.restore_pd_value(expected_predictions_df, loc)

    def test_prepare_tolerances(self):
        # make mock ValidationTestReader with just settings
        class Mock(object):
            def __init__(self):
                self.settings = None

        validation_test_reader = Mock()
        rel_tol, abs_tol = .1, .002
        validation_test_reader.settings = dict(relative=rel_tol, absolute=abs_tol)
        results_comparator = ResultsComparator(validation_test_reader, self.simulation_run_results)
        default_tolerances = ValidationUtilities.get_default_args(np.allclose)
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['rtol'], rel_tol)
        self.assertEqual(tolerances['atol'], abs_tol)
        validation_test_reader.settings['relative'] = None
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['rtol'], default_tolerances['rtol'])
        del validation_test_reader.settings['absolute']
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['atol'], default_tolerances['atol'])


class TestCaseValidator(unittest.TestCase):

    def setUp(self):
        self.test_case_num = '00101'
        self.test_case_num = '00001'
        self.case_validator = CaseValidator(TEST_CASES, 'discrete_stochastic', self.test_case_num)
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')

    def test_case_validator_errors(self):
        settings = self.case_validator.validation_test_reader.settings
        del settings['duration']
        with self.assertRaisesRegexp(ValidationError, "required setting .* not provided"):
            self.case_validator.validate_model()
        settings['duration'] = 'not a float'
        with self.assertRaisesRegexp(ValidationError, "required setting .* not a float"):
            self.case_validator.validate_model()
        settings['duration'] = 10.
        settings['start'] = 3
        with self.assertRaisesRegexp(ValidationError, "non-zero start setting .* not supported"):
            self.case_validator.validate_model()

    def make_plot_file(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        plot_file = os.path.join(self.tmp_dir, 'discrete_stochastic', "{}_{}.pdf".format(self.test_case_num, timestamp))
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        return plot_file

    def test_case_validator_discrete_stochastic(self):
        self.assertFalse(self.case_validator.validate_model())
        plot_file = self.make_plot_file()
        self.assertFalse(self.case_validator.validate_model(num_discrete_stochastic_runs=10,
            plot_file=plot_file))
        self.assertTrue(os.path.isfile(plot_file))

        # test validation failure
        expected_preds_df = self.case_validator.validation_test_reader.expected_predictions_df
        expected_preds_array = expected_preds_df.loc[:, 'X-mean'].values
        expected_preds_df.loc[:, 'X-mean'] = np.full(expected_preds_array.shape, 0)
        self.assertEqual(['X'], self.case_validator.validate_model(num_discrete_stochastic_runs=5,
            discard_run_results=False, plot_file=self.make_plot_file()))


class TestValidationUtilities(unittest.TestCase):

    def test_get_default_args(self):

        defaults = {'a': None,
            'b': 17,
            'c': frozenset(range(3))}
        def func(y, a=defaults['a'], b=defaults['b'], c=defaults['c']):
            pass
        self.assertEqual(defaults, ValidationUtilities.get_default_args(func))
