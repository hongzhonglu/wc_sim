#!/usr/bin/env python

import unittest
import sys
import re

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.SimulationEngine import SimulationEngine
from SequentialSimulator.multialgorithm.CellState import (Specie, CellState)
from SequentialSimulator.multialgorithm.MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
from UniversalSenderReceiverSimulationObject import UniversalSenderReceiverSimulationObject

class TestCellState(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    def test_invalid_event_types(self):
    
        # CellState( name, initial_population, debug=False ) 
        pop = dict( zip( 's1 s2 s3'.split(), range(3) ) )
        cs1 = CellState( 'name', pop, debug=False ) 
        # initial events
        with self.assertRaises(ValueError) as context:
            cs1.send_event( 1.0, cs1, 'init_msg1' )
        self.assertIn( "'CellState' simulation objects not registered to send 'init_msg1' messages", context.exception.message )

    id = 0
    @staticmethod
    def get_name():
        TestCellState.id += 1
        return "CellState_{:d}".format( TestCellState.id )
 
    species = 's1 s2 s3'.split()
    pop = dict( zip( species, map( lambda x: x*7, range(3,6) ) ) )

    @staticmethod
    def make_CellState( pop, debug=False, write_plot_output=False, name=None ):
        if not name:
            name = TestCellState.get_name()
        '''
        print "Creating CellState( {}, --population--, debug={}, write_plot_output={} ) ".format(
            name, debug, write_plot_output )
        '''
        return CellState( name, pop, debug=debug, write_plot_output=write_plot_output ) 
        
    def test_CellState_debugging(self):
        cs1 = TestCellState.make_CellState( TestCellState.pop, debug=False )
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr.send_event( 1.0, cs1, MessageTypes.GET_POPULATION )
        eq = cs1.event_queue_to_str()
        self.assertIn( 'CellState_1 at 0.000', eq )
        self.assertIn( 'creation_time\tevent_time\tsending_object\treceiving_object\tevent_type', eq )
        
        
    def test_simple_CellState(self):
        cs1 = TestCellState.make_CellState( TestCellState.pop, write_plot_output=False )
        # initial events
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr.send_event( 1.0, cs1, MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
            event_body=ADJUST_POPULATION_BY_DISCRETE_MODEL_body(
                dict( zip( TestCellState.species, [1]*len( TestCellState.species ) ) )
            )
        )
        t = 2.0
        d = dict( zip( TestCellState.species,
                map( lambda x: Continuous_change(2.0, 1.0), [1]*len( TestCellState.species ) ) ) )

        usr.send_event( t, cs1, MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
            event_body=ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body( d ) )
        SimulationEngine.simulate( 5.0 )
        
        # TODO(Arthur): test simultaneous recept of ADJUST_POPULATION_BY_DISCRETE_MODEL and ADJUST_POPULATION_BY_CONTINUOUS_MODEL

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

