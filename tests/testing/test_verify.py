"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-09-25
:Copyright: 2018, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
from collections import namedtuple
from inspect import currentframe, getframeinfo
from scipy.constants import Avogadro
import cProfile
import datetime
import math
import numpy as np
import os
import pandas
import pstats
import shutil
import tempfile
import unittest


from wc_sim.config import core as config_core_multialgorithm
from wc_sim.run_results import RunResults
from wc_sim.testing.verify import (VerificationError, VerificationTestCaseType, VerificationTestReader,
                                   ResultsComparator, ResultsComparator, CaseVerifier, VerificationResultType,
                                   VerificationSuite, VerificationUtilities, MultialgModelVerification,
                                   VerificationRunResult, ODETestIterators)
import obj_tables
import wc_lang
import wc_sim.submodels.odes as odes

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class TestODETestIterators(unittest.TestCase):

    def test_geometric_iterator(self):
        geometric_iterator = ODETestIterators.geometric_iterator
        self.assertEqual([2, 4, 8], list(geometric_iterator(2, 10, 2)))
        self.assertEqual([1e-05, 0.0001, 0.001, 0.01, 0.1], list(geometric_iterator(1E-5, 0.1, 10)))
        np.testing.assert_allclose([.1, .3], list(geometric_iterator(0.1, 0.3, 3)))
        with self.assertRaisesRegexp(ValueError, '0 < min is required'):
            next(geometric_iterator(-1, 0.3, 3))
        with self.assertRaisesRegexp(ValueError, '0 < min is required'):
            next(geometric_iterator(0, 0.3, 3))
        with self.assertRaisesRegexp(ValueError, 'min <= max is required'):
            next(geometric_iterator(1, 0.3, 3))
        with self.assertRaisesRegexp(ValueError, '1 < factor is required'):
            next(geometric_iterator(.1, 0.3, .6))

    def test_ode_test_iterator(self):
        ode_test_generator = ODETestIterators.ode_test_generator
        default_rtol = config_multialgorithm['rel_ode_solver_tolerance']
        default_atol = config_multialgorithm['abs_ode_solver_tolerance']
        self.assertEqual([{'ode_time_step_factor': 1.0, 'rtol': default_rtol, 'atol': default_atol}],
                         list(ode_test_generator()))

        def close_dicts(d1, d2):
            if set(d1.keys()) != set(d2.keys()):
                return False
            for k in d1.keys():
                if not math.isclose(d1[k], d2[k]):
                    return False
            return True

        min_rtol = 1E-10
        max_rtol = 1E-8
        rtol_range = dict(min=min_rtol, max=max_rtol)
        tolerance_ranges = {'rtol': rtol_range}
        generated_test_arguments = list(ode_test_generator(tolerance_ranges=tolerance_ranges))
        first_kwargs = generated_test_arguments[0]
        last_kwargs = generated_test_arguments[-1]
        self.assertTrue(close_dicts({'ode_time_step_factor': 1.0, 'rtol': min_rtol, 'atol': default_atol},
                        first_kwargs))
        self.assertTrue(close_dicts({'ode_time_step_factor': 1.0, 'rtol': max_rtol, 'atol': default_atol},
                        last_kwargs))

        min_atol = 1E-11
        max_atol = 1E-7
        atol_range = dict(min=min_atol, max=max_atol)
        tolerance_ranges = {'atol': atol_range}
        generated_test_arguments = list(ode_test_generator(tolerance_ranges=tolerance_ranges))
        first_kwargs = generated_test_arguments[0]
        last_kwargs = generated_test_arguments[-1]
        self.assertTrue(close_dicts({'ode_time_step_factor': 1.0, 'rtol': default_rtol, 'atol': min_atol},
                        first_kwargs))
        self.assertTrue(close_dicts({'ode_time_step_factor': 1.0, 'rtol': default_rtol, 'atol': max_atol},
                        last_kwargs))

        tolerance_ranges = {'rtol': rtol_range,
                            'atol': atol_range}
        generated_test_arguments = list(ode_test_generator(tolerance_ranges=tolerance_ranges))
        first_kwargs = generated_test_arguments[0]
        last_kwargs = generated_test_arguments[-1]
        self.assertTrue(close_dicts({'ode_time_step_factor': 1.0, 'rtol': min_rtol, 'atol': min_atol},
                        first_kwargs))
        self.assertTrue(close_dicts({'ode_time_step_factor': 1.0, 'rtol': max_rtol, 'atol': max_atol},
                        last_kwargs))

        ts_fcts = [.5, .1, 0.01]
        generated_test_arguments = list(ode_test_generator(ode_time_step_factors=ts_fcts,
                                                          tolerance_ranges=tolerance_ranges))
        first_kwargs = generated_test_arguments[0]
        last_kwargs = generated_test_arguments[-1]
        self.assertTrue(close_dicts({'ode_time_step_factor': ts_fcts[0], 'rtol': min_rtol, 'atol': min_atol},
                        first_kwargs))
        self.assertTrue(close_dicts({'ode_time_step_factor': ts_fcts[-1], 'rtol': max_rtol, 'atol': max_atol},
                        last_kwargs))


TEST_CASES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixtures',
                                          'verification', 'testing'))

def make_test_case_dir(test_case_num, test_case_type='DISCRETE_STOCHASTIC'):
    return os.path.join(TEST_CASES, VerificationTestCaseType[test_case_type].value, test_case_num)

def make_verification_test_reader(test_case_num, test_case_type):
    return VerificationTestReader(test_case_type, make_test_case_dir(test_case_num, test_case_type),
                                  test_case_num)


class TestVerificationTestReader(unittest.TestCase):

    def test_read_settings(self):
        settings = make_verification_test_reader('00001', 'DISCRETE_STOCHASTIC').read_settings()
        some_expected_settings = dict(
            start=0,
            variables=['X'],
            amount=['X'],
            sdRange=(-5, 5)
        )
        for expected_key, expected_value in some_expected_settings.items():
            self.assertEqual(settings[expected_key], expected_value)
        settings = make_verification_test_reader('00003', 'DISCRETE_STOCHASTIC').read_settings()
        self.assertEqual(settings['key1'], 'value1: has colon')
        self.assertEqual(
            make_verification_test_reader('00004', 'DISCRETE_STOCHASTIC').read_settings()['amount'], ['X', 'Y'])

        with self.assertRaisesRegexp(VerificationError,
            "duplicate key 'key1' in settings file '.*00002/00002-settings.txt"):
            make_verification_test_reader('00002', 'DISCRETE_STOCHASTIC').read_settings()

        with self.assertRaisesRegexp(VerificationError,
            "could not read settings file.*00005/00005-settings.txt.*No such file or directory.*"):
            make_verification_test_reader('00005', 'DISCRETE_STOCHASTIC').read_settings()

    def test_read_expected_predictions(self):
        for test_case_type in ['DISCRETE_STOCHASTIC', 'CONTINUOUS_DETERMINISTIC']:
            verification_test_reader = make_verification_test_reader('00001', test_case_type)
            verification_test_reader.settings = verification_test_reader.read_settings()
            expected_predictions_df = verification_test_reader.read_expected_predictions()
            self.assertTrue(isinstance(expected_predictions_df, pandas.core.frame.DataFrame))

        # wrong time sequence
        verification_test_reader.settings['duration'] += 1
        with self.assertRaisesRegexp(VerificationError,
                                     "times in settings .* differ from times in expected predictions"):
            verification_test_reader.read_expected_predictions()
        verification_test_reader.settings['duration'] -= 1

        # wrong columns
        missing_variable = 'MissingVariable'
        for test_case_type, expected_error in [
            ('DISCRETE_STOCHASTIC', "mean or sd of some amounts missing from expected predictions.*{}"),
            ('CONTINUOUS_DETERMINISTIC', "some amounts missing from expected predictions.*{}")]:
            verification_test_reader = make_verification_test_reader('00001', test_case_type)
            verification_test_reader.settings = verification_test_reader.read_settings()
            verification_test_reader.settings['amount'].append(missing_variable)
            with self.assertRaisesRegexp(VerificationError, expected_error.format(missing_variable)):
                verification_test_reader.read_expected_predictions()

    def test_slope_of_predictions(self):
        verification_test_reader = make_verification_test_reader('00001', 'CONTINUOUS_DETERMINISTIC')
        verification_test_reader.run()
        derivatives_df = verification_test_reader.slope_of_predictions()
        results = verification_test_reader.expected_predictions_df
        self.assertTrue(derivatives_df.time.equals(results.time))
        species = 'S1'
        self.assertEqual(derivatives_df.loc[0, species],
            (results.loc[1, species] - results.loc[0, species])/(results.time[1] - results.time[0]))

    def test_read_model(self):
        verification_test_reader = make_verification_test_reader('00001', 'DISCRETE_STOCHASTIC')
        model = verification_test_reader.read_model()
        self.assertTrue(isinstance(model, obj_tables.Model))
        self.assertEqual(model.id, 'test_case_' + verification_test_reader.test_case_num)

    def test_get_species_id(self):
        verification_test_reader = make_verification_test_reader('00001', 'DISCRETE_STOCHASTIC')
        verification_test_reader.run()
        self.assertEqual(verification_test_reader.get_species_id('X'), 'X[c]')

        # test exceptions
        with self.assertRaisesRegexp(VerificationError, "no species id found for species_type ''"):
            verification_test_reader.get_species_id('')

        # add a species with species type 'X' in another compartment
        model = verification_test_reader.model
        existing_compt = model.get_compartments()[0]
        new_compt = model.compartments.create(id='compt_new',
                                 biological_type=existing_compt.biological_type,
                                 init_volume=existing_compt.init_volume)
        existing_species_type = model.get_species_types()[0]
        new_species = new_compt.species.create(species_type=existing_species_type, model=model)
        new_species.id = new_species.gen_id()
        with self.assertRaisesRegexp(VerificationError, "multiple species ids for species_type X:"):
            verification_test_reader.get_species_id(existing_species_type.id)

    def test_verification_test_reader(self):
        test_case_num = '00001'
        verification_test_reader = make_verification_test_reader(test_case_num, 'DISCRETE_STOCHASTIC')
        self.assertEqual(None, verification_test_reader.run())

        # test __str__()
        self.assertIn('duration', str(verification_test_reader))
        self.assertIn('time', str(verification_test_reader))
        for attr_colon in ['expected_predictions_file:', 'model_filename:', 'settings_file:',
                           'test_case_dir:', 'test_case_num:', 'test_case_type:']:
            self.assertIn(attr_colon, str(verification_test_reader))

        # test exceptions
        with self.assertRaisesRegexp(VerificationError, "Unknown VerificationTestCaseType:"):
            VerificationTestReader('no_such_test_case_type', '', test_case_num)


# todo: fix
@unittest.skip('doesnt work for multiple compartments')
class TestResultsComparator(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.test_case_data = {
            'CONTINUOUS_DETERMINISTIC': {
                'test_case_num': '00001',
            },
            'DISCRETE_STOCHASTIC': {
                'test_case_num': '00001',
            }
        }
        for test_case_type, data in self.test_case_data.items():
            data['simulation_run_results'] = \
                self.make_run_results_from_expected_results(test_case_type, data['test_case_num'])
            data['verification_test_reader'] = \
                make_verification_test_reader(data['test_case_num'], test_case_type)
            data['verification_test_reader'].run()

    def make_run_results_hdf_filename(self):
        return os.path.join(tempfile.mkdtemp(dir=self.tmp_dir), RunResults.HDF5_FILENAME)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    # todo: add argument that maps species type ids to species ids
    def make_run_results(self, test_case_type, species_amount_df):
        # make a RunResults obj with given species amounts

        # create empty dfs for all RunResults components
        run_results_hdf_filename = self.make_run_results_hdf_filename()
        empty_df = pandas.DataFrame(index=[], columns=[])
        for component in RunResults.COMPONENTS:
            empty_df.to_hdf(run_results_hdf_filename, component)

        # add compartments to species
        cols = list(species_amount_df.columns)
        species_amount_df.columns = cols[:1] + list(map(ResultsComparator.get_species, cols[1:]))

        # species amounts are Moles for continuous (aka 'semantic') tests and copy numbers for stochastic tests
        if test_case_type == 'CONTINUOUS_DETERMINISTIC':
            # volumes in the continuous test cases are static at 1 liter
            VOLUMES = 1
            # for continuous tests, convert Moles to populations that will be stored in run results
            pops_df = species_amount_df * VOLUMES * Avogadro

        elif test_case_type == 'DISCRETE_STOCHASTIC':
            pops_df = species_amount_df

        # create an hdf
        pops_df.to_hdf(run_results_hdf_filename, 'populations')

        if test_case_type == 'CONTINUOUS_DETERMINISTIC':
            # create the compartment volumes, needed for continuous models
            columns = pandas.MultiIndex.from_tuples(((TEST_CASE_COMPARTMENT, 'volume'), ),
                                                    names=['compartment', 'property'])
            data = [VOLUMES] * len(pops_df.index)
            aggregate_states_df =  pandas.DataFrame(data=data, index=pops_df.index, columns=columns,
                                                    dtype=np.float64, copy=True)
            aggregate_states_df.to_hdf(run_results_hdf_filename, 'aggregate_states')

        # make & return a RunResults
        return RunResults(os.path.dirname(run_results_hdf_filename))

    def make_run_results_from_expected_results(self, test_case_type, test_case_num):
        """ Create a RunResults object with the same populations and volumes as a test case """
        results_file = os.path.join(TEST_CASES, VerificationTestCaseType[test_case_type],
                                    test_case_num, test_case_num+'-results.csv')
        results_df = pandas.read_csv(results_file)
        return self.make_run_results(test_case_type, results_df)

    def test_results_comparator_continuous_deterministic(self):
        test_case_data = self.test_case_data['CONTINUOUS_DETERMINISTIC']
        results_comparator = ResultsComparator(test_case_data['verification_test_reader'],
                                               test_case_data['simulation_run_results'])
        self.assertEqual(False, results_comparator.differs())

        # to fail a comparison, modify the run results for first species at time 0
        species_type_1 = test_case_data['verification_test_reader'].settings['amount'][0]
        species_1 = ResultsComparator.get_species(species_type_1)
        test_case_data['simulation_run_results'].get('populations').loc[0, species_1] *= 2
        self.assertTrue(results_comparator.differs())
        self.assertIn(species_type_1, results_comparator.differs())

        '''
        # todo: test functionality of tolerances
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

    def stash_pd_value(self, df, loc, new_val):
        self.stashed_pd_value = df.loc[loc[0], loc[1]]
        df.loc[loc[0], loc[1]] = new_val

    def restore_pd_value(self, df, loc):
        df.loc[loc[0], loc[1]] = self.stashed_pd_value

    def test_results_comparator_discrete_stochastic(self):
        test_case_type = 'DISCRETE_STOCHASTIC'
        test_case_data = self.test_case_data[test_case_type]

        # make multiple run_results with variable populations
        verification_test_reader = test_case_data['verification_test_reader']
        expected_predictions_df = verification_test_reader.expected_predictions_df
        times = expected_predictions_df.loc[:,'time'].values
        n_times = len(times)
        means = expected_predictions_df.loc[:,'X-mean'].values
        n_runs = 3
        run_results_list = []
        for i in range(n_runs):
            pops = np.empty((n_times, 2))
            pops[:,0] = times
            # add stochasticity to populations
            pops[:,1] = np.add(means, (np.random.rand(n_times)*2)-1)
            pop_df = pandas.DataFrame(pops, index=times, columns=['time', 'X'])
            run_results_list.append(self.make_run_results('DISCRETE_STOCHASTIC', pop_df))

        results_comparator = ResultsComparator(verification_test_reader, run_results_list)
        self.assertEqual(False, results_comparator.differs())

        ### adjust data to test all Z thresholds ###
        # test by altering data at this arbitrary time
        time = 1

        # Z                 expect differ?
        # -                 -------
        # range[0]-epsilon  yes
        # range[0]+epsilon  no
        # range[1]-epsilon  no
        # range[1]+epsilon  yes
        epsilon = 1e-9
        lower_range, upper_range = (verification_test_reader.settings['meanRange'][0],
                                    verification_test_reader.settings['meanRange'][1])
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
            # set all run results pops at time to pop_val
            species = ResultsComparator.get_species('X')
            for run_results in run_results_list:
                run_results.get('populations').loc[time, species] = pop_val

        for test_z_score, expected_differ in z_scores_and_expected_differs:
            test_pop_mean = get_test_pop_mean(time, n_runs, expected_predictions_df, test_z_score)
            set_all_pops(run_results_list, time, test_pop_mean)
            results_comparator = ResultsComparator(verification_test_reader, run_results_list)
            self.assertEqual(expected_differ, results_comparator.differs())

        loc = [0, 'X-mean']
        self.stash_pd_value(expected_predictions_df, loc, -1)
        with self.assertRaisesRegexp(VerificationError, "e_mean contains negative value.*"):
            ResultsComparator(verification_test_reader, run_results_list).differs()
        self.restore_pd_value(expected_predictions_df, loc)

        loc = [0, 'X-sd']
        self.stash_pd_value(expected_predictions_df, loc, -1)
        with self.assertRaisesRegexp(VerificationError, "e_sd contains negative value.*"):
            ResultsComparator(verification_test_reader, run_results_list).differs()
        self.restore_pd_value(expected_predictions_df, loc)

    def test_prepare_tolerances(self):
        # make mock VerificationTestReader with just settings
        class Mock(object):
            def __init__(self):
                self.settings = None

        verification_test_reader = Mock()
        rel_tol, abs_tol = .1, .002
        verification_test_reader.settings = dict(relative=rel_tol, absolute=abs_tol)
        test_case_data = self.test_case_data['CONTINUOUS_DETERMINISTIC']
        simulation_run_results = test_case_data['simulation_run_results']
        results_comparator = ResultsComparator(verification_test_reader, simulation_run_results)
        default_tolerances = VerificationUtilities.get_default_args(np.allclose)
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['rtol'], rel_tol)
        self.assertEqual(tolerances['atol'], abs_tol)
        verification_test_reader.settings['relative'] = None
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['rtol'], default_tolerances['rtol'])
        del verification_test_reader.settings['absolute']
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['atol'], default_tolerances['atol'])


class TestCaseVerifier(unittest.TestCase):

    def setUp(self):
        self.test_case_num = '00001'
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        self.case_verifiers = {}
        self.model_types = ['DISCRETE_STOCHASTIC', 'CONTINUOUS_DETERMINISTIC']
        for model_type in self.model_types:
            self.case_verifiers[model_type] = CaseVerifier(TEST_CASES, model_type, self.test_case_num,
                                                           default_num_stochastic_runs=10)

    def test_init_optional_args(self):
        case_verifier = CaseVerifier(TEST_CASES, 'DISCRETE_STOCHASTIC', self.test_case_num)
        self.assertEqual(case_verifier.default_num_stochastic_runs,
                         config_multialgorithm['num_ssa_verification_sim_runs'])
        num_runs = 10
        case_verifier = CaseVerifier(TEST_CASES, 'DISCRETE_STOCHASTIC', self.test_case_num,
                                     default_num_stochastic_runs=num_runs)
        self.assertEqual(case_verifier.default_num_stochastic_runs, num_runs)

    def test_case_verifier_errors(self):
        for model_type in self.model_types:
            settings = self.case_verifiers[model_type].verification_test_reader.settings
            del settings['duration']
            with self.assertRaisesRegexp(VerificationError, "required setting .* not provided"):
                self.case_verifiers[model_type].verify_model()
            settings['duration'] = 'not a float'
            with self.assertRaisesRegexp(VerificationError, "required setting .* not a float"):
                self.case_verifiers[model_type].verify_model()
            settings['duration'] = 10.
            settings['start'] = 3
            with self.assertRaisesRegexp(VerificationError, "non-zero start setting .* not supported"):
                self.case_verifiers[model_type].verify_model()

    def make_plot_file(self, model_type, case=None):
        if case is None:
            case = self.test_case_num
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
        plot_file = os.path.join(self.tmp_dir, model_type, "{}_{}.pdf".format(case, timestamp))
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        return plot_file

    def test_case_verifier(self):
        for model_type in self.model_types:
            plot_file = self.make_plot_file(model_type)
            self.assertFalse(self.case_verifiers[model_type].verify_model(plot_file=plot_file))
            self.assertTrue(os.path.isfile(plot_file))

    def test_verify_model_stochastic_fails(self):
        # test a failure despite running CaseVerifier.MAX_TRIES
        discrete_verifier = self.case_verifiers['DISCRETE_STOCHASTIC']
        # alter the expected predictions
        expected_predictions_df = discrete_verifier.verification_test_reader.expected_predictions_df
        expected_predictions_df.loc[:, 'X-mean'] *= 2
        comparison_result = discrete_verifier.verify_model(num_discrete_stochastic_runs=3)
        self.assertTrue(comparison_result)

    def test_verify_model_optional_args(self):
        discrete_verifier = self.case_verifiers['DISCRETE_STOCHASTIC']
        continuous_verifier = self.case_verifiers['CONTINUOUS_DETERMINISTIC']

        # test num_discrete_stochastic_runs
        comparison_result = discrete_verifier.verify_model(num_discrete_stochastic_runs=3)
        self.assertFalse(comparison_result)
        comparison_result = discrete_verifier.verify_model(num_discrete_stochastic_runs=0)
        self.assertFalse(comparison_result)

        # test discard_run_results
        comparison_result = continuous_verifier.verify_model(discard_run_results=True)
        self.assertFalse(comparison_result)
        self.assertFalse(os.path.isdir(continuous_verifier.tmp_results_dir))
        comparison_result = continuous_verifier.verify_model(discard_run_results=False)
        self.assertFalse(comparison_result)
        self.assertTrue(os.path.isdir(continuous_verifier.tmp_results_dir))

        # test ode_time_step_factor
        ode_time_step_factor = 1.0
        comparison_result = continuous_verifier.verify_model(ode_time_step_factor=ode_time_step_factor)
        self.assertFalse(comparison_result)

        # test tolerances
        test_atol = 1E-10
        test_rtol = 1E-10
        tolerances = dict(atol=test_atol, rtol=test_rtol)
        comparison_result = continuous_verifier.verify_model(tolerances=tolerances)
        self.assertFalse(comparison_result)

    def test_plot_model_verification(self):
        discrete_verifier = self.case_verifiers['DISCRETE_STOCHASTIC']
        for presentation_qual in [True, False]:
            plot_file = self.make_plot_file('DISCRETE_STOCHASTIC')
            comparison_result = discrete_verifier.verify_model(num_discrete_stochastic_runs=5)
            discrete_verifier.plot_model_verification(plot_file, presentation_qual=presentation_qual)
            self.assertTrue(os.path.isfile(plot_file))
            # alter the expected predictions, so second plot shows failed verification
            expected_predictions_df = discrete_verifier.verification_test_reader.expected_predictions_df
            expected_predictions_df.loc[:, 'X-mean'] *= 2


class TestVerificationSuite(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.verification_suite = VerificationSuite(TEST_CASES, self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_init(self):
        self.assertEqual(self.verification_suite.plot_dir, self.tmp_dir)
        no_such_dir = os.path.join(self.tmp_dir, 'no_such_dir')
        with self.assertRaisesRegexp(VerificationError, "cannot open cases_dir"):
            VerificationSuite(no_such_dir)
        with self.assertRaisesRegexp(VerificationError, "cannot open plot_dir"):
            VerificationSuite(TEST_CASES, no_such_dir)

    def test_record_result(self):
        self.assertEqual(self.verification_suite.results, [])
        sub_dir = os.path.join(self.tmp_dir, 'test_case_sub_dir')

        result = VerificationRunResult(TEST_CASES, sub_dir, '00001', VerificationResultType.CASE_VERIFIED,
                                       1.1, output='test output', error='eg error')
        self.verification_suite._record_result(*result[1:])
        self.assertEqual(len(self.verification_suite.results), 1)
        self.assertEqual(self.verification_suite.results[-1], result)

        result = VerificationRunResult(TEST_CASES, sub_dir, '00001',
                                       VerificationResultType.CASE_UNREADABLE, 0)
        self.verification_suite._record_result(*result[1:])
        self.assertEqual(len(self.verification_suite.results), 2)
        self.assertEqual(self.verification_suite.results[-1], result)

        with self.assertRaisesRegexp(VerificationError,
                                     "result_type must be a VerificationResultType, not a"):
            self.verification_suite._record_result(TEST_CASES, sub_dir, '00001',
                                                   'not a VerificationResultType')

    def test__run_test(self):
        test_case_num = '00001'
        self.verification_suite._run_test('CONTINUOUS_DETERMINISTIC', test_case_num)
        results = self.verification_suite.get_results()
        self.assertEqual(results.pop().result_type, VerificationResultType.CASE_VERIFIED)
        plot_file_name_prefix = 'CONTINUOUS_DETERMINISTIC' + '_' + test_case_num
        self.assertIn(plot_file_name_prefix, os.listdir(self.tmp_dir).pop())

        # test without plotting
        verification_suite = VerificationSuite(TEST_CASES)
        verification_suite._run_test('CONTINUOUS_DETERMINISTIC', test_case_num)
        self.assertEqual(verification_suite.results.pop().result_type,
                         VerificationResultType.CASE_VERIFIED)

        # be verbose
        verification_suite = VerificationSuite(TEST_CASES)
        with CaptureOutput(relay=False) as capturer:
            verification_suite._run_test('CONTINUOUS_DETERMINISTIC', test_case_num, verbose=True)
            self.assertIn('Verifying CONTINUOUS_DETERMINISTIC case 00001', capturer.get_text())

        # rtol or atol, but not both
        default_rtol = config_multialgorithm['rel_ode_solver_tolerance']
        self.verification_suite._run_test('CONTINUOUS_DETERMINISTIC', test_case_num, rtol=default_rtol)
        self.assertEqual(results.pop().result_type, VerificationResultType.CASE_VERIFIED)

        default_atol = config_multialgorithm['abs_ode_solver_tolerance']
        self.verification_suite._run_test('CONTINUOUS_DETERMINISTIC', test_case_num, atol=default_atol)
        self.assertEqual(results.pop().result_type, VerificationResultType.CASE_VERIFIED)

        # case unreadable
        verification_suite = VerificationSuite(TEST_CASES)
        self.verification_suite._run_test('invalid_case_type_name', test_case_num)
        self.assertEqual(results.pop().result_type, VerificationResultType.CASE_UNREADABLE)

        # run fails
        plot_dir = tempfile.mkdtemp()
        verification_suite = VerificationSuite(TEST_CASES, plot_dir)
        # delete plot_dir to create failure
        shutil.rmtree(plot_dir)
        verification_suite._run_test('DISCRETE_STOCHASTIC', test_case_num, num_stochastic_runs=2)
        self.assertEqual(verification_suite.results.pop().result_type,
                         VerificationResultType.FAILED_VERIFICATION_RUN)

        verification_suite = VerificationSuite(TEST_CASES)
        verification_suite._run_test('DISCRETE_STOCHASTIC', '00006', num_stochastic_runs=1)
        self.assertEqual(verification_suite.results.pop().result_type,
                         VerificationResultType.CASE_DID_NOT_VERIFY)

    def test__run_tests(self):
        ode_time_step_factors = [.5, 1, 5]
        results = self.verification_suite._run_tests('CONTINUOUS_DETERMINISTIC', '00001',
                                                     ode_time_step_factors=ode_time_step_factors)
        self.assertEqual(len(results), len(ode_time_step_factors))
        last_result = results[-1]
        params = eval(last_result.output)
        self.assertEqual(params['ode_time_step_factor'], ode_time_step_factors[-1])

        max_rtol = 1E-9
        max_atol = 1E-11
        test_tolerance_ranges = {'rtol': dict(min=1E-10, max=max_rtol),
                                 'atol': dict(min=1E-13, max=max_atol)}
        results = self.verification_suite._run_tests('CONTINUOUS_DETERMINISTIC', '00001',
                                                     tolerance_ranges=test_tolerance_ranges,
                                                     empty_results=True)
        self.assertEqual(len(results), 2 * 3)
        last_result = results[-1]
        params = eval(last_result.output)
        self.assertEqual(params['tolerances']['rtol'], max_rtol)
        self.assertEqual(params['tolerances']['atol'], max_atol)

        results = self.verification_suite._run_tests('CONTINUOUS_DETERMINISTIC', '00001',
                                                     ode_time_step_factors=ode_time_step_factors,
                                                     tolerance_ranges=test_tolerance_ranges,
                                                     empty_results=True)
        self.assertEqual(len(results), len(ode_time_step_factors) * 2 * 3)
        last_result = results[-1]
        params = eval(last_result.output)
        self.assertEqual(params['ode_time_step_factor'], ode_time_step_factors[-1])
        self.assertEqual(params['tolerances']['rtol'], max_rtol)
        self.assertEqual(params['tolerances']['atol'], max_atol)

    def test_tolerance_ranges_for_sensitivity_analysis(self):
        tolerance_ranges = VerificationSuite.tolerance_ranges_for_sensitivity_analysis()
        for tol in ['rtol', 'atol']:
            self.assertIn(tol, tolerance_ranges)
            for limit in ['min', 'max']:
                self.assertIn(limit, tolerance_ranges[tol])

    def test_run(self):
        results = self.verification_suite.run('CONTINUOUS_DETERMINISTIC', ['00001'])
        self.assertEqual(results[-1].result_type, VerificationResultType.CASE_VERIFIED)

        results = self.verification_suite.run('CONTINUOUS_DETERMINISTIC', ['00001', '00006'])
        expected_types = [VerificationResultType.CASE_VERIFIED, VerificationResultType.CASE_UNREADABLE]
        result_types = [result_tuple.result_type for result_tuple in results[-2:]]
        self.assertEqual(expected_types, result_types)

        results = self.verification_suite.run('DISCRETE_STOCHASTIC', num_stochastic_runs=5)
        self.assertEqual(expected_types, result_types)

        results = self.verification_suite.run(num_stochastic_runs=5)
        self.assertEqual(expected_types, result_types)

        with self.assertRaisesRegexp(VerificationError, 'cases should be an iterator over case nums'):
            self.verification_suite.run('DISCRETE_STOCHASTIC', '00001')

        with self.assertRaisesRegexp(VerificationError, 'if cases provided then test_case_type_name must'):
            self.verification_suite.run(cases=['00001'])

        with self.assertRaisesRegexp(VerificationError, 'Unknown VerificationTestCaseType: '):
            self.verification_suite.run(test_case_type_name='no such VerificationTestCaseType')


SsaTestCase = namedtuple('SsaTestCase', 'case_num, dsmts_num, num_ssa_runs')


class RunVerificationSuite(unittest.TestCase):

    def setUp(self):
        NUM_SIMULATION_RUNS = 20
        self.ssa_test_cases = [
            # see: https://github.com/sbmlteam/sbml-test-suite/blob/master/cases/stochastic/DSMTS-userguide-31v2.pdf
            SsaTestCase('00001', '001-01', NUM_SIMULATION_RUNS),
            SsaTestCase('00003', '001-03', NUM_SIMULATION_RUNS),
            SsaTestCase('00004', '001-04', NUM_SIMULATION_RUNS),
            SsaTestCase('00007', '001-07', 2 * NUM_SIMULATION_RUNS),
            SsaTestCase('00012', '001-12', NUM_SIMULATION_RUNS),
            SsaTestCase('00020', '002-01', NUM_SIMULATION_RUNS),
            SsaTestCase('00021', '002-02', NUM_SIMULATION_RUNS),
            SsaTestCase('00030', '003-01', NUM_SIMULATION_RUNS),
            SsaTestCase('00037', '004-01', 4 * NUM_SIMULATION_RUNS)
        ]
        self.ode_test_cases = [
            '00001',
            '00002',
            '00003',
            '00004',
            '00005',
            '00006',
            '00010',
            '00014',
            '00015',
            '00017',
            '00018',
            '00019',
            '00020',
            '00021',
            '00022',
            '00028',
            '00054', # 2 compartments
        ]
        self.multialgorithmic_test_cases = [
            '00007',
            '00030',
        ]
        self.root_test_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'fixtures',
                                              'verification', 'cases'))
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.plot_dir = os.path.join(os.path.dirname(__file__), 'tmp', 'verification', timestamp)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.verification_suite = VerificationSuite(self.root_test_dir, self.plot_dir)

    def run_verification_cases(self, case_type, verification_cases, testing=False):

        if case_type == 'DISCRETE_STOCHASTIC':
            for ssa_test_case in verification_cases:
                self.verification_suite.run(case_type, [ssa_test_case.case_num],
                                            num_stochastic_runs=ssa_test_case.num_ssa_runs, verbose=True)

        if case_type == 'CONTINUOUS_DETERMINISTIC':
            for ode_test_case in verification_cases:
                self.verification_suite.run(case_type, [ode_test_case], verbose=True)

        failures = []
        successes = []
        for result in self.verification_suite.get_results():
            if result.error:
                failure_msg = "{} {}\n".format(result.case_num, result.result_type.name) + \
                    "{}".format(result.error)
                failures.append(failure_msg)
            else:
                successes.append("{} {}".format(result.case_num, result.result_type.name))
        if testing:
            self.assertTrue(failures == [], msg='\n'.join(successes + failures))
        if successes:
            print('\nsuccess(es):\n' + '\n'.join(successes))
        if failures:
            msg = "SBML test suite case(s) failed validation:\n" + '\n'.join(failures)
            self.fail(msg=msg)
        return self.verification_suite.get_results(), failures, successes

    def test_verification_stochastic(self):
        # todo: move to verification main program
        results, _, _ = self.run_verification_cases('DISCRETE_STOCHASTIC', self.ssa_test_cases)

    def test_verification_deterministic(self):
        # todo: set good tolerances for SBML test cases
        # abs_ode_solver_tolerance = 1e-10
        # rel_ode_solver_tolerance = 1e-8
        self.run_verification_cases('CONTINUOUS_DETERMINISTIC', self.ode_test_cases)

    @unittest.skip('Fails on first 150 SBML models in the test suite')
    def test_convert_sbml_to_wc_lang(self):
        # try to use the wc_lang SBML Importer to create wc_lang test models
        # fails on the first 150 models in the SBML test suite - their components lack sufficient units
        from wc_lang.sbml import io as sbml_io
        from wc_lang.sbml.util import LibSbmlInterface
        from wc_lang.io import Reader, Writer
        import libsbml
        import glob

        # convert the SBML level 3 version 1 and 2 models to wc lang
        # write a converted wc lang model xlsx file in the same directory
        errors = []
        model_types = ['semantic', 'stochastic']
        for model_type in model_types:
            model_type_dir = os.path.join(self.root_test_dir, model_type)
            print(f'processing: {model_type_dir}')
            sbml_models = glob.glob(os.path.join(model_type_dir, '0*/0*-sbml-l3v[1-2].xml'))
            for sbml_model in sorted(sbml_models):
                print(f'sbml_model: {os.path.basename(sbml_model)}')
                sbml_model_dir = os.path.dirname(sbml_model)
                basename_parts = os.path.basename(sbml_model).split('-')
                test_num = basename_parts[0]
                sbml_lvl_ver = basename_parts[2].split('.')[0]
                wc_lang_model_filename = f'{test_num}-wc_conv-{sbml_lvl_ver}.xlsx'
                wc_lang_model_pathname = os.path.join(sbml_model_dir, wc_lang_model_filename)

                sbml_reader = LibSbmlInterface.call_libsbml(libsbml.SBMLReader)
                sbml_doc = LibSbmlInterface.call_libsbml(sbml_reader.readSBMLFromFile, sbml_model)
                LibSbmlInterface.raise_if_error(sbml_doc, f'Model could not be read from {sbml_model}')
                try:
                    wc_lang_model = sbml_io.SbmlImporter().run(sbml_doc)
                    print(f'writing: {wc_lang_model_filename}')
                    Writer().run(wc_lang_model_pathname, wc_lang_model, data_repo_metadata=False,
                                 protected=False)
                except wc_lang.sbml.util.LibSbmlError as e:
                    errors.append(f"could not read '{sbml_model}: {e}'")
        print('Could not process\n', '\n'.join(errors))

    def test_verification_hybrid(self):
        # transcription_translation_case = SsaTestCase('transcription_translation', 'NA', 10)
        # translation_metabolism_case = SsaTestCase('translation_metabolism', 'NA', 10)
        results, _, _ = self.run_verification_cases('MULTIALGORITHMIC', self.multialgorithmic_test_cases)
        return
        profile = False
        if profile:
            print('profiling')
            tmp_dir = tempfile.mkdtemp()
            out_file = os.path.join(tmp_dir, "profile_out.out")
            locals = {'self': self,
                      'test_case': test_case}
            cProfile.runctx("self.run_verification_cases('DISCRETE_STOCHASTIC', [test_case])",
                            {}, locals, filename=out_file)
            profile = pstats.Stats(out_file)
            print("Profile for 'test_case' simulation objects")
            profile.strip_dirs().sort_stats('cumulative').print_stats(20)
        else:
            self.run_verification_cases('DISCRETE_STOCHASTIC', [test_case])


class TestVerificationUtilities(unittest.TestCase):

    def test_get_default_args(self):

        defaults = {'a': None,
            'b': 17,
            'c': frozenset(range(3))}
        def func(y, a=defaults['a'], b=defaults['b'], c=defaults['c']):
            pass
        self.assertEqual(defaults, VerificationUtilities.get_default_args(func))


@unittest.skip('incomplete')
class TestHybridModelVerification(unittest.TestCase):

    HYBRID_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'verification', 'testing', 'hybrid')

    def setUp(self):
        self.hybrid_model_verifications = {}

        trans_trans_hybrid_settings = dict(
            start='0',
            duration='2.88E+04',
            steps='2.88E+02',
            amount='RNA, protein',
            output='RNA-mean, RNA-sd, protein-mean, protein-sd',
            meanRange='(-3, 3)',
        )
        trans_trans_correct_ssa_settings = trans_trans_hybrid_settings
        self.trans_trans_model_base = 'transcription_translation'
        self.hybrid_model_verifications[self.trans_trans_model_base] = \
            self.make_hybrid_verification(self.trans_trans_model_base, trans_trans_hybrid_settings,
                trans_trans_correct_ssa_settings)

        trans_met_hybrid_settings = dict(
            start='0',
            duration='2.88E+04',
            steps='2.88E+02',
            amount='protein, Ala, H2O',
            output='protein-mean, protein-sd, Ala-mean, Ala-sd, H2O-mean, H2O-sd',
            meanRange='(-3, 3)',
        )
        trans_met_correct_ssa_settings = trans_met_hybrid_settings
        self.trans_met_model_base = 'translation_metabolism'
        self.hybrid_model_verifications[self.trans_met_model_base] = \
            self.make_hybrid_verification(self.trans_met_model_base, trans_met_hybrid_settings,
                trans_met_correct_ssa_settings)

    def tearDown(self):
        pass

    def make_multialg_verification(self, model_base_filename, correct_ssa_settings, hybrid_settings):
        verification_dir = tempfile.mkdtemp(dir=self.HYBRID_DIR)
        correct_ssa_model_file = os.path.join(self.HYBRID_DIR, model_base_filename + '_correct_ssa.xlsx')
        hybrid_model_file = os.path.join(self.HYBRID_DIR, model_base_filename + '_hybrid.xlsx')
        multialg_model_verification = MultialgModelVerificationFuture(
            verification_dir,
            model_base_filename,
            correct_ssa_model_file,
            correct_ssa_settings,
            hybrid_model_file,
            hybrid_settings
        )
        return multialg_model_verification

    def test(self):
        # in [self.trans_trans_model_base, self.trans_met_model_base]:
        for model_base_name in [self.trans_trans_model_base, ]:
            self.hybrid_model_verifications[self.trans_trans_model_base].run()
