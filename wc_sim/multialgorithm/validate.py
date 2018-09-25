"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-09-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import warnings
from enum import Enum
import pandas as pd
import numpy as np

from wc_lang.io import Reader
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
    

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


class ModelReader(object):
    """ Read a model

    Read a model into a `wc_lang` representation. Read SBML or `wc_lang` models.

    Attributes:
        model_filename (:obj:`str`): the model's filename
    """
    SBML_FILE_SUFFIX = '.xml'

    def __init__(self, model_filename):
        """
        Args:
            model_filename (:obj:`str`): the model's filename
        """
        if model_filename.endswith(self.SBML_FILE_SUFFIX):
            raise MultialgorithmError("Reading SBML files not supported: model filename '{}'".
                format(model_filename))
        self.model_filename = model_filename
        

    def run(self):
        '''
            1. read the model
            2. if SBML, convert to wc_lang
        '''
        return Reader().run(self.model_filename, strict=False)        


class ConfigReader(object):
    """ Read a validation test case configuration """
    '''
        1. read simulation configuration
        2. read validation test configuration
    '''
    pass


class ValidationTestCaseType(Enum):
    """ Types of test cases """
    continuous_deterministic = 1    # algorithms like ODE
    discrete_stochastic = 2         # algorithms like SSA


class ValidationTestReader(object):
    """ Read a model validation test case """
    def __init__(self, test_case_type, test_case_dir, test_case_num):
        if test_case_type not in ValidationTestCaseType.__members__:
            raise ValidationError("Unknown ValidationTestCaseType: '{}'".format(test_case_type))
        else:
            self.test_case_type = ValidationTestCaseType[test_case_type]
        if self.test_case_type == ValidationTestCaseType.continuous_deterministic:
            raise ValidationError('not implemented')
        if self.test_case_type == ValidationTestCaseType.discrete_stochastic:
            pass
        self.test_case_dir = test_case_dir
        self.test_case_num = test_case_num

    def read_model(self):
        pass

    def read_expected_predictions(self):
        if self.test_case_type == ValidationTestCaseType.continuous_deterministic:
            pass
        if self.test_case_type == ValidationTestCaseType.discrete_stochastic:
            expected_predictions
            return expected_predictions
        #pd.read_csv()

    def read_settings(self):
        """ Read a test case's settings into a key-value dictionary """
        self.settings_file = settings_file = os.path.join(self.test_case_dir, self.test_case_num+'-settings.txt')
        settings = {}
        errors = []
        try:
            with open(settings_file, 'r') as f:
                for line in f:
                    key, value = line.strip().split(':', maxsplit=1)
                    if key in settings:
                        errors.append("duplicate key '{}' in settings file '{}'".format(key, settings_file))
                    settings[key] = value.strip()
        except Exception as e:
            errors.append("could not read settings file '{}': {}".format(settings_file, e))
        if errors:
            raise ValidationError('; '.join(errors))
        return settings

    def run():
        self.settings = self.read_settings()


class ResultsComparator(object):
    """ Compare simulated and expected predictions """
    pass


class ResultsComparator(object):
    """ Compare simulated and expected predictions """
    pass


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
            deterministic: if simulated trajectory == expected trajectory: success
            stochastic:
                evaluate whether mean of simulation trajectories match expected trajectory
                    run # simulations executions
                    compare mean of simulated trajectories with expected trajectory
                    use XXX to evaluate whether mean is close to expected trajectory, express as a p-value
                    if P[mean of trajectories differs from expected trajectory] > p-value threshold: failure
                if failure, retry "evaluate whether mean of simulation trajectories match expected trajectory"
                    # simulations generating the correct trajectories will fail validation (100*(p-value threshold)) percent of the time
                if failure again, report failure    # assuming p-value << 1, two failures indicates likely errors
    '''
    pass


class ValidationSuite(object):
    """ Run suite of validation tests of `wc_sim`'s dynamic behavior """
    pass
