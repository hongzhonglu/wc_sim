""" Test RunResults

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import shutil
import tempfile
import pandas
import os
import numpy
from capturer import CaptureOutput
import pytest

from wc_lang.core import SpeciesType
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults
from wc_sim.multialgorithm.make_models import MakeModels


class TestRunResults2(unittest.TestCase):

    def setUp(self):
        # create stored checkpoints and metadata
        self.temp_dir = tempfile.mkdtemp()
        # create and run ODE simulation
        self.model_file = os.path.join(os.path.dirname(__file__), 'fixtures', 'validation', 'testing',
            'semantic', '00001', '00001-wc_lang.xlsx')
        simulation = Simulation(self.model_file)
        self.checkpoint_period = 0.1
        self.max_time = 1
        _, self.results_dir = simulation.run(end_time=self.max_time, results_dir=self.temp_dir,
            checkpoint_period=self.checkpoint_period, time_step=self.checkpoint_period)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_convert_to_concentrations(self):
        run_results = RunResults(self.results_dir)
        # todo: make this a test
        print("run_results.get('concentrations')", run_results.get('concentrations'))

    def test_exceptions(self):
        # put restoration of RunResults.COMPUTED_COMPONENTS in finally so it always runs
        tmp_COMPUTED_COMPONENTS = RunResults.COMPUTED_COMPONENTS
        RunResults.COMPUTED_COMPONENTS = {'test': 'not_a_func'}
        try:
            # todo: make this report a failure
            with self.assertRaisesRegexp(MultialgorithmError,
                'in COMPUTED_COMPONENTS is not a function in'):
                RunResults.prepare_computed_components()
        except:
            pass
        finally:
            RunResults.COMPUTED_COMPONENTS = tmp_COMPUTED_COMPONENTS


@unittest.skip('broken')
class TestRunResults(unittest.TestCase):

    def setUp(self):
        # create stored checkpoints and metadata
        self.temp_dir = tempfile.mkdtemp()
        # create and run simulation
        model = MakeModels.make_test_model('2 species, 1 reaction')
        simulation = Simulation(model)
        self.checkpoint_period = 10
        self.max_time = 100
        with CaptureOutput(relay=False) as capturer:
            _, self.results_dir = simulation.run(end_time=self.max_time, results_dir=self.temp_dir,
                checkpoint_period=self.checkpoint_period)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @pytest.mark.timeout(15)
    def test_run_results(self):

        run_results_1 = RunResults(self.results_dir)
        # after run_results file created
        run_results_2 = RunResults(self.results_dir)
        for component in RunResults.COMPONENTS:
            component_data = run_results_1.get(component)
            self.assertTrue(run_results_1.get(component).equals(run_results_2.get(component)))

        expected_times = pandas.Float64Index(numpy.linspace(0, self.max_time, 1 + self.max_time/self.checkpoint_period))
        for component in ['populations', 'observables', 'aggregate_states', 'random_states']:
            component_data = run_results_1.get(component)
            self.assertFalse(component_data.empty)
            self.assertTrue(component_data.index.equals(expected_times))

        # total population is invariant
        populations = run_results_1.get('populations')
        pop_sum = populations.sum(axis='columns')
        for time in expected_times:
            self.assertEqual(pop_sum[time], pop_sum[0.])

        metadata = run_results_1.get('metadata')
        self.assertEqual(metadata['simulation']['time_max'], self.max_time)

    @pytest.mark.timeout(15)
    def test_run_results_errors(self):

        run_results = RunResults(self.results_dir)
        with self.assertRaisesRegexp(MultialgorithmError, "component '.*' is not an element of "):
            run_results.get('not_a_component')
