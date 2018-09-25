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

from wc_sim.multialgorithm.validate import (ValidationError, SubmodelValidator, SsaValidator, FbaValidator,
    OdeValidator, ModelReader, ConfigReader, ValidationTestCaseType, ValidationTestReader, ResultsComparator,
    ResultsComparator, ValidationTestRunner, ValidationSuite)


class TestValidate(unittest.TestCase):

    def setUp(self):
        self.TEST_CASES = os.path.join(os.path.dirname(__file__), 'fixtures', 'validation', 'test_cases')

    def tearDown(self):
        pass

    def make_test_reader(self, test_case_num):
        test_case_dir = os.path.join(self.TEST_CASES, test_case_num)
        return ValidationTestReader('discrete_stochastic', test_case_dir, test_case_num)

    def test_make_test_reader(self):
        test_reader = self.make_test_reader('00001')

    def test_test_reader_read_settings(self):
        test_reader = self.make_test_reader('00001')
        settings = test_reader.read_settings()
        expected_settings = dict(
            key1='value1 is long',
            key2='value2'
        )
        self.assertEqual(settings, expected_settings)
        settings = self.make_test_reader('00003').read_settings()
        self.assertEqual(settings['key1'], 'value1: has colon')

        with self.assertRaisesRegexp(ValidationError,
            "duplicate key 'key1' in settings file '.*test_cases/00002/00002-settings.txt"):
            self.make_test_reader('00002').read_settings()

        with self.assertRaisesRegexp(ValidationError,
            "could not read settings file.*00005/00005-settings.txt.*No such file or directory.*"):
            self.make_test_reader('00005').read_settings()
