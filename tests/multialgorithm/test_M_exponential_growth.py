""" Test exponential growth behavior in a multialgorithm simulation

:Author: Andrew Sundstrom (aes@acm.org)
:Date: 16 Aug 2018
:Copyright: 2018, Karr Lab
:License: MIT
"""


import warnings

from capturer import CaptureOutput
from numpy import diagonal, exp, linspace, mean, sqrt
from os import path
from pandas import Float64Index
from scipy.constants import Avogadro
from scipy.optimize import curve_fit
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase

from wc_lang.io import Reader
from wc_lang.prepare import PrepareModel, CheckModel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.run_results import RunResults
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel


MODEL_FILENAME = path.join(path.dirname(__file__), 'fixtures', 'test_model_for_exponential_growth_in_M.xlsx')


def build_dynamic_submodels(model):
    dynamic_submodels = {}
    for submodel in model.get_submodels():
            multialgorithm_simulation = MultialgorithmSimulation(model, {})
            multialgorithm_simulation.build_simulation()
            dynamic_submodels[submodel.id] = DynamicSubmodel(
                submodel.id,
                multialgorithm_simulation.dynamic_model,
                submodel.reactions,
                submodel.get_species(), 
                model.get_parameters(),
                multialgorithm_simulation.get_dynamic_compartments(submodel), 
                multialgorithm_simulation.local_species_population)
    return dynamic_submodels


def get_dynamic_submodel_data(dynamic_submodel):
    params = dynamic_submodel.get_parameter_values()
    counts = dynamic_submodel.get_specie_counts()
    rates = dynamic_submodel.calc_reaction_rates()
    reactions = dynamic_submodel.reactions
    return params, counts, rates, reactions


class TestInitialRate_dM(TestCase):

    def setUp(self):
        self.model = Reader().run(MODEL_FILENAME, strict=False)
        PrepareModel(self.model).run()
        CheckModel(self.model).run()
        self.dynamic_submodels = build_dynamic_submodels(self.model)
    
    def test_calc_reaction_rates_dM(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            params, counts, rates, reactions = get_dynamic_submodel_data(dynamic_submodel)
            for rxn_index, rxn in enumerate(reactions):
                if rxn.id == 'dM':
                    self.assertEquals(rates[rxn_index], params['growthRate']*counts['M[c]'])
                    

class TestExponentialGrowth_M(TestCase):

    def setUp(self):
        self.model = Reader().run(MODEL_FILENAME, strict=False)
        self.dynamic_submodels = build_dynamic_submodels(self.model)
        self.temp_dir = mkdtemp()
        simulation = Simulation(self.model)
        self.checkpoint_period = 60
        self.end_time = 108000
        with CaptureOutput(relay=False) as capturer:
            _, self.results_dir = simulation.run(
                end_time=self.end_time, 
                results_dir=self.temp_dir,
                checkpoint_period=self.checkpoint_period)
    
    def tearDown(self):
        rmtree(self.temp_dir)

    def test_exponential_growth_in_M(self):
        dynamic_submodel = self.dynamic_submodels['cell_cycle']
        params, counts, rates, reactions = get_dynamic_submodel_data(dynamic_submodel)
        time = Float64Index(linspace(0, self.end_time, 1 + self.end_time/self.checkpoint_period))
        run_results = RunResults(self.results_dir)
        populations = run_results.get('populations')
        M_abundance = populations['M[c]']
        cf_par, cf_cov = curve_fit(
            lambda t,a,b: a*exp(b*t),
            time,
            M_abundance,
            p0=(M_abundance[time[0]], 1e-6))
        cf_M_0 = cf_par[0]
        cf_exp = cf_par[1]
        self.assertTrue(abs(cf_M_0 - counts['M[c]']) < counts['M[c]']/100)
        self.assertTrue(abs(cf_exp - params['growthRate']) < params['growthRate']/10)
        cf_err = sqrt(diagonal(cf_cov))
        cf_err_M_0 = cf_err[0]
        cf_err_exp = cf_err[1]
        self.assertTrue(cf_err_M_0 < counts['M[c]']/1000)
        self.assertTrue(cf_err_exp < params['growthRate']/100)
        sse, sst = 0, 0
        mean_M_abundance = mean(M_abundance)
        for t in time:
            cf_func_val = cf_M_0 * exp(cf_exp*t)
            sqe = (cf_func_val - M_abundance[t])**2
            sqt = (cf_func_val - mean_M_abundance)**2
            sse += sqe
            sst += sqt
        rsq = 1 - (sse/sst)
        self.assertTrue(rsq > 0.9999)
