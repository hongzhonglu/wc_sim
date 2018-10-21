"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-10-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import os
import numpy as np
import scikits

from wc_lang.core import SubmodelAlgorithm, CompartmentType
from wc_sim.multialgorithm.submodels.odes import OdeSubmodel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.message_types import RunOde
from wc_sim.multialgorithm.make_models import MakeModels
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_expressions import DynamicRateLaw


class TestOdeSubmodel(unittest.TestCase):

    def setUp(self):
        self.ode_submodel_empty = OdeSubmodel('test_1', None, [], [], [], [], None, 1)
        self.mdl_1_spec = MakeModels.make_test_model('1 species, 1 reaction')
        self.ode_sbmdl_1_spec = self.make_ode_submodel(self.mdl_1_spec)

    def make_ode_submodel(self, model, time_step=1.0):
        """Make a MultialgorithmSimulation from a wc lang model"""
        # assume a single submodel
        # todo: have MakeModels accept algorithm types
        model.submodels[0].algorithm = SubmodelAlgorithm.ode
        self.time_step = time_step
        args = dict(time_step=self.time_step)
        multialgorithm_simulation = MultialgorithmSimulation(model, args)
        simulation, dynamic_model = multialgorithm_simulation.build_simulation()
        return multialgorithm_simulation.simulation_submodels[0]

    # test low level methods
    def test_ode_submodel_init(self):
        self.assertEqual(self.ode_sbmdl_1_spec.time_step, self.time_step)
        bad_time_step = -2
        with self.assertRaisesRegexp(MultialgorithmError,
            'time_step must be positive, but is {}'.format(bad_time_step)):
            self.make_ode_submodel(self.mdl_1_spec, time_step=bad_time_step)

    def test_set_up_optimizations(self):
        ode_submodel = self.ode_sbmdl_1_spec
        self.assertTrue(set(ode_submodel.species_ids) == ode_submodel.species_ids_set \
            == set(ode_submodel.adjustments.keys()))
        for pre_alloc_array in [ode_submodel.concentrations, ode_submodel.populations]:
            self.assertEqual(pre_alloc_array.shape, (len(ode_submodel.species_ids), ))

    # todo: for the next 4 tests, check results against raw properties of self.mdl_1_spec
    def test_get_compartment_id(self):
        self.assertEqual(self.ode_sbmdl_1_spec.get_compartment_id(), 'compt_1')

    def test_get_compartment_volume(self):
        self.assertEqual(self.ode_sbmdl_1_spec.get_compartment_volume(), 1e-16)

    def test_get_concentrations(self):
        self.assertEqual(self.ode_sbmdl_1_spec.get_concentrations(), np.array([0.01660539040427164]))

    def test_concentrations_to_populations(self):
        self.assertEqual(self.ode_sbmdl_1_spec.concentrations_to_populations(
            self.ode_sbmdl_1_spec.concentrations), np.array([0.]))

    def test_solver_lock(self):
        self.assertTrue(self.ode_submodel_empty.get_solver_lock())
        with self.assertRaisesRegexp(MultialgorithmError, 'OdeSubmodel .*: cannot get_solver_lock'):
            self.ode_submodel_empty.get_solver_lock()
        self.assertTrue(self.ode_submodel_empty.release_solver_lock())

    # test solving
    def test_set_up_ode_submodel(self):
        # todo: test for model with multiple species, and species in multiple reactions
        self.ode_sbmdl_1_spec.set_up_ode_submodel()
        self.assertEqual(self.ode_sbmdl_1_spec, OdeSubmodel.instance)

        rate_of_change_expressions = self.ode_sbmdl_1_spec.rate_of_change_expressions
        # model has 1 species: spec_type_0[compt_1]
        self.assertEqual(len(rate_of_change_expressions), 1)
        # spec_type_0[compt_1] participates in 1 reaction
        self.assertEqual(len(rate_of_change_expressions[0]), 1)
        # the reaction consumes 1 spec_type_0[compt_1]
        coeffs, rate_law = rate_of_change_expressions[0][0]
        self.assertEqual(coeffs, -1)
        self.assertEqual(type(rate_law), DynamicRateLaw)

        # test exception: remove the rate laws
        self.mdl_1_spec.get_reactions()[0].rate_laws = []
        with self.assertRaisesRegexp(MultialgorithmError,
            "OdeSubmodel .* dynamic reaction .* needs dynamic rate law"):
            self.make_ode_submodel(self.mdl_1_spec)

    def test_set_up_ode_solver(self):
        solver_return = self.ode_sbmdl_1_spec.set_up_ode_solver()
        self.assertTrue(solver_return.flag)
        self.assertEqual(type(self.ode_sbmdl_1_spec.solver), scikits.odes.ode)

    def test_right_hand_side(self):
        time = 0
        concentrations = self.ode_sbmdl_1_spec.get_concentrations()
        concentration_change_rates = np.zeros(len(self.ode_sbmdl_1_spec.concentrations))
        flag = OdeSubmodel.right_hand_side(time, concentrations, concentration_change_rates)
        self.assertEqual(flag, 0)

        # create failure by changing rate_of_change_expressions
        # turn testing off
        self.ode_sbmdl_1_spec.testing = False
        self.ode_sbmdl_1_spec.rate_of_change_expressions[-1] = None
        self.assertEqual(OdeSubmodel.right_hand_side(time, concentrations, concentration_change_rates), 1)
        # turn testing on
        self.ode_sbmdl_1_spec.testing = True
        with self.assertRaisesRegexp(MultialgorithmError, "OdeSubmodel .* solver.right_hand_side.* failed"):
            OdeSubmodel.right_hand_side(time, concentrations, concentration_change_rates)

    def test_run_ode_solver(self):
        # todo
        pass

    # test event scheduling and handling
    def test_schedule_next_ode_analysis(self):
        custom_time_step = 4
        custom_ode_submodel = self.make_ode_submodel(self.mdl_1_spec, time_step=custom_time_step)
        # no events are scheduled
        self.assertTrue(custom_ode_submodel.simulator.event_queue.empty())

        # check that the next event is a RunOde message at time expected_time
        def check_next_event(expected_time):
            next_event = custom_ode_submodel.simulator.event_queue.next_events()[0]
            self.assertEqual(next_event.creation_time, 0)
            self.assertEqual(next_event.event_time, expected_time)
            self.assertEqual(next_event.sending_object, custom_ode_submodel)
            self.assertEqual(next_event.receiving_object, custom_ode_submodel)
            self.assertEqual(type(next_event.message), RunOde)
            self.assertTrue(custom_ode_submodel.simulator.event_queue.empty())

        # initial event should be at 0
        custom_ode_submodel.send_initial_events()
        check_next_event(0)

        # next RunOde event should be at custom_time_step
        custom_ode_submodel.schedule_next_ode_analysis()
        check_next_event(custom_time_step)

    def test_handle_RunOde_msg(self):
        pass
