""" Test multialgorithm main program

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-05-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import re
import shutil
import tempfile
import unittest
from argparse import Namespace
from capturer import CaptureOutput
from copy import copy
from wc_lang.core import SpeciesType
from wc_sim import __main__
from wc_sim.multialgorithm.__main__ import SimController


class SimControllerTestCase(unittest.TestCase):
    def setUp(self):
        SpeciesType.objects.reset()
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           '2_species_1_reaction.xlsx')
        self.checkpoints_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.checkpoints_dir)

    def test_parse_args(self):
        args = Namespace(
            model_file='wc_lang_model.xlsx',
            end_time=100,
            checkpoint_period=3,
            checkpoints_dir=self.checkpoints_dir,
            dataframe_file=os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            fba_time_step=5.5
        )
        original_args = copy(args)
        SimController.process_and_validate_args(args)
        for arg, value in args.__dict__.items():
            self.assertEqual(getattr(original_args, arg), value)

        args.dataframe_file = os.path.join(self.checkpoints_dir, 'dataframe_file_no_suffix')
        original_args = copy(args)
        SimController.process_and_validate_args(args)
        self.assertEqual(args.dataframe_file, original_args.dataframe_file + '.h5')

        # test error detection
        errors = dict(
            end_time=[-3, 0],
            checkpoint_period=[-2, 0, args.end_time + 1],
            fba_time_step=[-2, 0, args.end_time + 1],
        )
        for arg, error_vals in errors.items():
            for error_val in error_vals:
                args2 = copy(args)
                setattr(args2, arg, error_val)
                with self.assertRaises(ValueError):
                    SimController.process_and_validate_args(args2)

        # test dataframe_file requires checkpoints_dir
        args = Namespace(
            model_file='wc_lang_model.xlsx',
            end_time=100,
            dataframe_file='dataframe_file.h5',
            checkpoint_period=1,
            checkpoints_dir=None,
            fba_time_step=10,
        )
        with self.assertRaisesRegexp(ValueError,
            'dataframe_file cannot be specified unless checkpoints_dir is provided'):
            SimController.process_and_validate_args(args)

    def test_simulate(self):
        argv = [
            'sim',
            self.MODEL_FILENAME,
            '10',
            '--checkpoint-period', '3',
            '--checkpoints-dir', self.checkpoints_dir,
            '--dataframe-file', os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            '--fba-time-step', '5',
        ]
        with __main__.App(argv=argv) as app:
            with CaptureOutput() as capturer:
                app.run()
                events = re.search('^Simulated (\d+) events', capturer.get_text())
                checkpoints = re.search("Saved chcekpoints in '(.*?)'$", capturer.get_text())

        num_events = int(events.group(1))
        res_dirname = checkpoints.group(1)
        # TODO(Arthur): stronger assertions
        self.assertTrue(0 < num_events)
        self.assertTrue(res_dirname.startswith(self.checkpoints_dir))