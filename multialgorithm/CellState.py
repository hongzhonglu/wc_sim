"""
A species counts cell state encapsulated in a SimulationObject.

Created 2016/07/19
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""
import sys
import logging

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
from Sequential_WC_Simulator.core.SimulationEngine import MessageTypesRegistry
from specie import Specie

    
class CellState( SimulationObject ): 
    """The cell's state, which represents the state of its species.
    
    CellState tracks the population of a set of species. After initialization, population values 
    are incremented or decremented, not written. To enable multi-algorithmic modeling, 
    it supports adjustments to a specie's population by both discrete and continuous models. E.g., discrete models
    might synthesize proteins while a continuous model degrades them.
    
    Attributes:
        population: a set species, represented by Specie objects
        debug: whether to log debugging data
        logger_name: 
    
    Event messages:

        ADJUST_POPULATION_BY_DISCRETE_MODEL
        ADJUST_POPULATION_BY_CONTINUOUS_MODEL
        GET_POPULATION
        
        See message descriptions in MessageTypes.
        At any time instant, ADJUST_POPULATION_* has priority over GET_POPULATION
        
    # TODO(Arthur): extend beyond population, e.g., to represent binding sites for individual macromolecules
    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): abstract the functionality of CellState so it can be integrated into a submodel
    # TODO(Arthur): report error if a Specie instance is updated by multiple continuous sub-models
    """
    
    SENT_MESSAGE_TYPES = [ MessageTypes.GIVE_POPULATION ]
    # TODO(Arthur): can Python automatically get the object name (e.g. 'CellState') here?
    MessageTypesRegistry.set_sent_message_types( 'CellState', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
        MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
        MessageTypes.GET_POPULATION ]

    MessageTypesRegistry.set_receiver_priorities( 'CellState', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, name, initial_population, initial_fluxes=None, 
        shared_random_seed=None, debug=False, write_plot_output=False, log=False):
        """Initialize a CellState object.
        
        Initialize a CellState object. Establish its initial population, and set debugging booleans.
        
        Args:
            name: string; name of this simulation object
            initial_population: initial population for some species;
                dict: specie_name -> initial_population
                species that are not initialized in this argument are assumed to be initialized with no population
                # TODO(Arthur): evalaute whether this is a good assumption
            initial_fluxes: optional; dict: specie_name -> initial_flux; 
                initial fluxes for all species whose populations are estimated by a continuous model
                fluxes are ignored for species not specified in initial_population
            shared_random_seed: hashable object; optional; set to deterministically initialize all random number
                generators used by the Specie objects
            debug: boolean; log debugging output
            write_plot_output: boolean; log output for plotting simulation 
            log: boolean; log population dynamics of species

        Raises:
            AssertionError: if the population cannot be initialized.
        """
        super( CellState, self ).__init__( name, plot_output=write_plot_output )

        self.population = {}
        try:
            for specie_name in initial_population.keys():
                initial_flux_given = None
                if initial_fluxes is not None and specie_name in initial_fluxes:
                    initial_flux_given = initial_fluxes[specie_name]
                self.population[specie_name] = Specie( specie_name, initial_population[specie_name], 
                    initial_flux=initial_flux_given, random_seed=shared_random_seed )
        except AssertionError as e:
            sys.stderr.write( "Cannot initialize CellState: {}.\n".format( e.message ) )
            
        self.logger_name = "CellState_{}".format( name )
        if debug:
            # make a logger for this CellState
            # TODO(Arthur): eventually control logging when creating SimulationObjects, and pass in the logger
            setup_logger( self.logger_name, level=logging.DEBUG )
            mylog = logging.getLogger(self.logger_name)
            # write initialization data
            mylog.debug( "initial_population: {}".format( str(initial_population) ) )
            mylog.debug( "initial_fluxes: {}".format( str(initial_fluxes) ) )
            mylog.debug( "shared_random_seed: {}".format( str(shared_random_seed) ) )
            mylog.debug( "write_plot_output: {}".format( str(write_plot_output) ) )
            mylog.debug( "debug: {}".format( str(debug) ) )
            mylog.debug( "log: {}".format( str(log) ) )

        # if log, make a logger for each specie
        my_level = logging.NOTSET
        if log:
            my_level = logging.DEBUG
        for specie_name in initial_population.keys():
            setup_logger(specie_name, level=my_level )
            log = logging.getLogger(specie_name)
            # write log header
            log.debug( '\t'.join( 'Sim_time Adjustment_type New_population New_flux'.split() ) )
            # log initial state
            self.log_event( 'initial_state', self.population[specie_name] )


    def handle_event( self, event_list ):
        """Handle a CellState simulation event.
        
        If an event message adjusts the population of a specie that is not known, initialize the Specie with
        no population. 
        
        # TODO(Arthur): More desc. 
        
        Args:
            event_list: list of event messages to process
            
        Raises:
            ValueError: if a GET_POPULATION message requests the population of an unknown species
            ValueError: if an ADJUST_POPULATION_BY_CONTINUOUS_MODEL event acts on a non-existent species

        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        super( CellState, self ).handle_event( event_list )
        
        logging.getLogger( self.logger_name ).debug( self.event_queue_to_str() ) 

        for event_message in event_list:
            # switch/case on event message type
            if event_message.event_type == MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL:

                # population_changes is an ADJUST_POPULATION_BY_DISCRETE_MODEL_body object
                population_changes = event_message.event_body

                logging.getLogger( self.logger_name ).debug( 
                    "ADJUST_POPULATION_BY_DISCRETE_MODEL: {}".format( str(population_changes) ) ) 
                for specie_name in population_changes.population_change.keys():
                    if not specie_name in self.population:
                        self.population[specie_name] = Specie( specie_name, 0 )
                    
                    self.population[specie_name].discrete_adjustment( 
                        population_changes.population_change[specie_name] )
                    self.log_event( 'discrete_adjustment', self.population[specie_name] )
                    
            elif event_message.event_type == MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL:
            
                # population_changes is an ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body object
                population_changes = event_message.event_body
                logging.getLogger( self.logger_name ).debug( 
                    "ADJUST_POPULATION_BY_CONTINUOUS_MODEL: {}".format( str(population_changes) ) ) 

                for specie_name in population_changes.population_change.keys():
                    # raise exeception if an ADJUST_POPULATION_BY_CONTINUOUS_MODEL event acts on a 
                    # non-existent species because the species has no flux, and we don't want a default flux
                    if not specie_name in self.population:
                        raise ValueError( "Error: {} message requests population of unknown species '{}' in {}\n".format(
                            MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, specie_name, event_message ) )
                    
                    #(population_change, flux) = population_changes[specie_name]
                    self.population[specie_name].continuous_adjustment( 
                        population_changes.population_change[specie_name].change, 
                        self.time, 
                        population_changes.population_change[specie_name].flux )
                    self.log_event( 'continuous_adjustment', self.population[specie_name] )

            elif event_message.event_type == MessageTypes.GET_POPULATION:

                # species is a GET_POPULATION_body object
                species = event_message.event_body
                # detect species requested that are not stored by this CellState
                invalid_species = species.species - set( self.population.keys() )
                # TODO(Arthur): test this case:                
                if len( invalid_species ):
                    raise ValueError( "Error: {} message requests population of unknown species {} in {}\n".format(
                        MessageTypes.GET_POPULATION,
                        str( list( invalid_species ) ),
                        str( event_message ) ) )

                # use Specie() function for determining population
                reported_population = {}
                for specie in species.species:
                    # always provide the current simulation time to get_population() because it is needed if the 
                    # specie has been updated by a continuous model
                    population = self.population[specie].get_population( time=self.time ) 
                    reported_population[ specie ] = population
                self.send_event( 0, event_message.sending_object, 
                    MessageTypes.GIVE_POPULATION, event_body=GIVE_POPULATION_body(reported_population) )

            else:
                assert False, "Shouldn't get here - event_message.event_type should be covered in the "
                "if statement above"

    def log_event( self, event_type, specie ):
        """Log simulation events that modify a specie's population.
        
        Log the simulation time, event type, specie population, and current flux for each simulation
        event message that adjusts the population. The log record is written to a separate log for each specie.
        
        Args:
            event_type: string; description of the event's type
            specie: Specie object; the object whose adjustment is being logged
        """
        log = logging.getLogger( specie.specie_name )
        try:
            flux = specie.continuous_flux
        except AttributeError:
            flux = None
        values = [ self.time, event_type, specie.last_population, flux ]
        values = map( lambda x: str(x), values )
        # log Sim_time Adjustment_type New_population New_flux
        log.debug( '\t'.join( values ) )
