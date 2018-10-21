""" Store and retrieve combined results of a multialgorithmic simulation run

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""
import numpy
import os
import pandas
import pickle
import warnings
from scipy.constants import Avogadro

from wc_utils.util.misc import as_dict
from wc_sim.log.checkpoint import Checkpoint
from wc_sim.core.sim_metadata import SimulationMetadata
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError


# todo: have hdf use disk more efficiently; for unknown reasons, in one example
# hdf uses 4.4M to store the data in 52 pickle files of 8K each
# the attempt with to_hdf_kwargs does not help
class RunResults(object):
    """ Store and retrieve combined results of a multialgorithmic simulation run

    Attributes:
        results_dir (:obj:`str`): pathname of a directory containing a simulation run's checkpoints and/or
            HDF5 file storing the combined results
        run_results (:obj:`dict`): dictionary of RunResults components, indexed by component name
    """
    # component stored in a RunResults instance and the HDF file it manages
    COMPONENTS = {
        # predicted populations of species at all checkpoint times
        'populations',
        # predicted aggregate states of the cell over the simulation
        'aggregate_states',
        # predicted values of all observables over the simulation
        'observables',
        # states of the simulation's random number geerators over the simulation
        'random_states',
        # the simulation's global metadata
        'metadata',
    }
    COMPUTED_COMPONENTS = {
        # predicted concentrations of species at all checkpoint times
        'concentrations': 'get_concentrations',
    }
    HDF5_FILENAME = 'run_results.h5'

    def __init__(self, results_dir):
        """ Create a `RunResults`

        Args:
            results_dir (:obj:`str`): directory storing checkpoints and/or HDF5 file with
                the simulation run results
        """
        self.results_dir = results_dir
        self.run_results = {}

        # if the HDF file containing the run results exists, open it
        if os.path.isfile(self._hdf_file()):
            self._load_hdf_file()

        # else create the HDF file from the stored metadata and sequence of checkpoints
        else:
            to_hdf_kwargs = dict(complevel=9, complib='blosc:zstd')

            # create the HDF file containing the run results
            population_df, observables_df, aggregate_states_df, random_states_s = self.convert_checkpoints()
            # populations
            population_df.to_hdf(self._hdf_file(), 'populations', **to_hdf_kwargs)
            # observables
            observables_df.to_hdf(self._hdf_file(), 'observables', **to_hdf_kwargs)
            # aggregate states
            aggregate_states_df.to_hdf(self._hdf_file(), 'aggregate_states', **to_hdf_kwargs)
            # todo (Arthur): address performance warning raised by pandas/io/pytables for the next two to_hdf() calls:
            '''
                /usr/local/lib/python3.6/site-packages/pandas/core/generic.py:1996 PerformanceWarning:
                your performance may suffer as PyTables will pickle object types that it cannot
                map directly to c-types [inferred_type->mixed,key->values] [items->None]
            '''
            # Temporarily, these warnings are being suppressed by the 'with warnings.catch_warnings()' context manager
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # random states
                random_states_s.to_hdf(self._hdf_file(), 'random_states', **to_hdf_kwargs)

                # metadata
                metadata_s = self.convert_metadata()
                metadata_s.to_hdf(self._hdf_file(), 'metadata', **to_hdf_kwargs)

            self._load_hdf_file()

    @classmethod
    def prepare_computed_components(cls):
        # check and initialize the COMPUTED_COMPONENTS
        for component, function in cls.COMPUTED_COMPONENTS.items():
            if hasattr(cls, function):
                cls.COMPUTED_COMPONENTS[component] = getattr(cls, function)
            else:
                raise MultialgorithmError("'{}' in COMPUTED_COMPONENTS is not a function in {}".format(
                    function, cls.__class__.__name__))

    def _hdf_file(self):
        """ Provide the pathname of the HDF5 file storing the combined results

        Returns:
            :obj:`str`: the pathname of the HDF5 file storing the combined results
        """
        return os.path.join(self.results_dir, self.HDF5_FILENAME)

    def _load_hdf_file(self):
        """ Load run results from the HDF file
        """
        try:
            for component in self.COMPONENTS:
                self.run_results[component] = pandas.read_hdf(self._hdf_file(), component)
        except Exception as e:
            # raise this exception because Pytables, used by pandas for hdf I/O, doesn't
            # reliably report 'No space left on device'
            raise MultialgorithmError("Unable to read hdf file: exception {}: disk may be full".format(str(e)))

    def get(self, component):
        """ Read and provide the specified `component`

        Args:
            component (:obj:`str`): the name of the component to return

        Returns:
            :obj:`pandas.DataFrame`, or `pandas.Series`: a pandas object containing a component of
                this `RunResults`, as specified by `component`

        Raises:
            :obj:`MultialgorithmError`: if `component` is not an element of `RunResults.COMPONENTS`
        """
        if component not in RunResults.COMPONENTS.union(RunResults.COMPUTED_COMPONENTS):
            raise MultialgorithmError("component '{}' is not an element of {}".format(component,
                RunResults.COMPONENTS))
        if component in RunResults.COMPUTED_COMPONENTS:
            return RunResults.COMPUTED_COMPONENTS[component](self)
        return self.run_results[component]

    def get_concentrations(self):
        """ Get concentrations

        Returns:
            :obj:`pandas.DataFrame`: the concentrations of this `RunResults`' species
        """
        # todo: return concentrations in all compartments
        # return self.get('populations') / (Avogadro * self.get('aggregate_states').loc[:, ('c', 'volume')])
        # todo: make sure this works by using volumes != 1
        volumes = self.get('aggregate_states').loc[:, ('c', 'volume')]
        return self.get('populations').iloc[:,:].div(volumes, axis=0) / Avogadro

    def convert_metadata(self):
        """ Convert the saved simulation metadata into a pandas series

        Returns:
            :obj:`pandas.Series`: the simulation metadata
        """
        simulation_metadata = SimulationMetadata.read_metadata(self.results_dir)
        return pandas.Series(as_dict(simulation_metadata))

    @staticmethod
    def get_state_components(state):
        return (state['population'], state['observables'], state['aggregate_state'])

    def convert_checkpoints(self):
        """ Convert the data in saved checkpoints into pandas dataframes

        Returns:
            :obj:`tuple` of pandas objects: dataframes of the components of a simulation checkpoint history
                population_df, observables_df, aggregate_states_df, random_states_s
        """
        # create pandas objects for species populations, aggregate states and simulation random states
        checkpoints = Checkpoint.list_checkpoints(self.results_dir)
        first_checkpoint = Checkpoint.get_checkpoint(self.results_dir, time=0)
        species_pop, observables, aggregate_state = self.get_state_components(first_checkpoint.state)

        species_ids = species_pop.keys()
        population_df = pandas.DataFrame(index=checkpoints, columns=species_ids, dtype=numpy.float64)

        observable_ids = observables.keys()
        observables_df = pandas.DataFrame(index=checkpoints, columns=observable_ids, dtype=numpy.float64)

        compartments = list(aggregate_state['compartments'].keys())
        properties = list(aggregate_state['compartments'][compartments[0]].keys())
        compartment_property_tuples = list(zip(compartments, properties))
        columns = pandas.MultiIndex.from_tuples(compartment_property_tuples, names=['compartment', 'property'])
        aggregate_states_df = pandas.DataFrame(index=checkpoints, columns=columns)
        random_states_s = pandas.Series(index=checkpoints)

        # load these pandas objects
        for time in Checkpoint.list_checkpoints(self.results_dir):

            checkpoint = Checkpoint.get_checkpoint(self.results_dir, time=time)
            species_populations, observables, aggregate_state = self.get_state_components(checkpoint.state)

            for species_id,population in species_populations.items():
                population_df.loc[time, species_id] = population

            for observable_id, observable in observables.items():
                observables_df.loc[time, observable_id] = observable

            compartment_states = aggregate_state['compartments']
            for compartment_id,agg_states in compartment_states.items():
                for property,value in agg_states.items():
                    aggregate_states_df.loc[time, (compartment_id, property)] = value

            random_states_s[time] = pickle.dumps(checkpoint.random_state)

        return (population_df, observables_df, aggregate_states_df, random_states_s)

# check and initialize the COMPUTED_COMPONENTS
RunResults.prepare_computed_components()