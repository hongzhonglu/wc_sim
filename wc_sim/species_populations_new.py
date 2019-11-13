""" Store species populations, and partition them among submodel private species and shared species

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-04
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

# TODO(Arthur): for reproducibility, use lists instead of sets
# TODO(Arthur): analyze accuracy with and without interpolation

import abc
import numpy
import sys
from collections import defaultdict
from scipy.constants import Avogadro
import math
from collections import namedtuple
from enum import Enum

import wc_lang
from de_sim.simulation_object import (SimulationObject, ApplicationSimulationObject,
                                           AppSimObjAndABCMeta, ApplicationSimulationObjMeta)
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.debug_logs import logs as debug_logs
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import NegativePopulationError, SpeciesPopulationError
from wc_sim import distributed_properties
from wc_utils.util.dict import DictUtil
from wc_utils.util.rand import RandomStateManager


class AccessSpeciesPopulationInterface(metaclass=abc.ABCMeta):   # pragma: no cover; methods in ABCs aren't run
    """ An abstract base class defining the interface between a submodel and its species population store(s)

    A submodel in a WC simulation can interact with multiple components that store the population
    of the species it models. This architecture is needed to simulate a model in parallel. All these
    stores must
    implement this interface which defines read and write operations on the species in a store.
    Both write operations have the prefix `adjust` in their names because they adjust a store's population.
    All operations require a time argument that indicates the simulation time at which the operation
    executes in the store.
    """

    @abc.abstractmethod
    def read_one(self, time, species_id):
        """ Obtain the predicted population of a specie at a particular simulation time """
        raise NotImplemented

    @abc.abstractmethod
    def read(self, time, species):
        """ Obtain the predicted population of a list of species at a particular simulation time """
        raise NotImplemented

    @abc.abstractmethod
    def adjust_discretely(self, time, adjustments):
        """ A discrete submodel adjusts the population of a set of species at a particular simulation time """
        raise NotImplemented

    @abc.abstractmethod
    def adjust_continuously(self, time, adjustments):
        """ A continuous submodel adjusts the population of a set of species at a particular simulation time """
        raise NotImplemented


 config_multialgorithm = \
     config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']
 LOCAL_POP_STORE = 'LOCAL_POP_STORE'  # the name of the local population store
 
 
 # TODO(Arthur): cover after MVP wc_sim done
 class AccessSpeciesPopulations(AccessSpeciesPopulationInterface):   # pragma: no cover
     """ Interface a submodel with the components that store the species populations it models
 
     Each submodel is a distinct simulation object. In the current population-based model,
     species are represented by their populations. (A hybrid instance-population model would change
     that.) Each submodel accesses a subset of the species in a model. A submodel's
     species can be partitioned into those that are accessed ONLY by the submodel and those that it
     shares with other submodels. These are respectively stored in a local LocalSpeciesPopulation
     which is private to this submodel, and a set of SpeciesPopSimObjects which are shared with other
     submodels. LocalSpeciesPopulation objects are accessed via local memory operations whereas
     SpeciesPopSimObjects, which are distinct simulation objects, are accessed via simulation event
     messages.
 
     AccessSpeciesPopulations enables a submodel to access all of the species populations that it
     uses through a single convenient interface. The submodel simply indicates the specie(s)
     being used and the operation type. This object then maps the specie(s) to the entity or entities
     storing them, and executes the operation on each entity.
 
     Essentially, an AccessSpeciesPopulations multiplexes a submodel's access to multiple population
     stores.
 
     Attributes:
-        submodel (:obj:`DynamicSubmodel`): the submodel which is using this AccessSpeciesPopulations
+        submodel (:obj:`DynamicSubmodel`): the submodel that's using this `AccessSpeciesPopulations`
         species_locations (:obj:`dict` of `str`): a map indicating the store for each specie used
             by the submodel using this object, that is, the local submodel.
         local_pop_store (:obj:`LocalSpeciesPopulation`): a store of local species
         remote_pop_stores (:obj:`dict` of identifiers of `SpeciesPopSimObject`): a map from store name
             to a system identifier for the remote population store(s) that the local submodel uses.
             For a shared memory implementation system identifiers can be object references; for a
             distributed implementation they must be network object identifiers.
         species_population_cache (:obj:`SpeciesPopulationCache`): a cache of populations for species
             that are stored remotely in the SpeciesPopSimObjects in remote_pop_stores; values for
             remote populations are pre-fetched at the right simulation time (via GetPopulation and
             GivePopulation messages) into this cache and then read from it when needed.
     """
-
     def __init__(self, local_pop_store, remote_pop_stores):
         """ Initialize an AccessSpeciesPopulations object
 
         The submodel object referenced in the attribute submodel must reference this
         AccessSpeciesPopulations instance. This object must be instantiated first; then the submodel
         can be created and set its reference with the set_*() method below.
 
         Raises:
             :obj:`SpeciesPopulationError`: if the remote_pop_stores contains a store named
                 'LOCAL_POP_STORE', which is a reserved store identifier for the local_pop_store.
         """
         self.local_pop_store = local_pop_store
         if LOCAL_POP_STORE in remote_pop_stores:
             raise SpeciesPopulationError("AccessSpeciesPopulations.__init__: {} not a valid "
-                                         "remote_pop_store name".format(LOCAL_POP_STORE))
+                "remote_pop_store name".format(LOCAL_POP_STORE))
         self.remote_pop_stores = remote_pop_stores
         self.species_locations = {}
         self.species_population_cache = SpeciesPopulationCache(self)
 
     def set_submodel(self, submodel):
         """ Set the submodel that uses this AccessSpeciesPopulations """
         self.submodel = submodel
 
-    def add_species_locations(self, store_name, species_ids, replace=False):
+    def add_species_locations(self, store_name, specie_ids, replace=False):
         """ Add species locations to the species location map
 
         Record that the species listed in `species_ids` are stored by the species population store
         identified by `store_name`. To replace existing location map values without raising an
         exception, set `replace` to True.
 
         Args:
             store_name (:obj:`str`): the globally unique name of a species population store. `LOCAL_POP_STORE`
                 is a special name that identifies the local population store for private species
-            species_ids (:obj:`list` of :obj:`str`): a list of species ids
+            specie_ids (:obj:`list` of `str`): a list of species ids
 
         Raises:
             :obj:`SpeciesPopulationError`: if store `store_name` is unknown
-            :obj:`SpeciesPopulationError`: if `replace` is False and any species_id in `species_ids` is
+            :obj:`SpeciesPopulationError`: if `replace` is False and any specie_id in `specie_ids` is
                 already mapped to a different store than `store_name`.
         """
         if not store_name in self.remote_pop_stores.keys() and store_name != LOCAL_POP_STORE:
             raise SpeciesPopulationError("add_species_locations: '{}' not a known population "
-                                         "store.".format(store_name))
+                "store.".format(store_name))
         if replace:
-            for species_id in species_ids:
-                self.species_locations[species_id] = store_name
+            for specie_id in specie_ids:
+                self.species_locations[specie_id] = store_name
         else:
-            assigned = list(filter(lambda s: s in self.species_locations.keys(), species_ids))
+            assigned = list(filter(lambda s: s in self.species_locations.keys(), specie_ids))
             if assigned:
                 raise SpeciesPopulationError("add_species_locations: species {} already have assigned "
-                                             "locations.".format(sorted(assigned)))
-            for species_id in species_ids:
-                self.species_locations[species_id] = store_name
+                    "locations.".format(sorted(assigned)))
+            for specie_id in specie_ids:
+                self.species_locations[specie_id] = store_name
 
-    def del_species_locations(self, species_ids, force=False):
+    def del_species_locations(self, specie_ids, force=False):
         """ Delete entries from the species location map
 
-        Remove species location mappings for the species in `species_ids`. To avoid raising an
+        Remove species location mappings for the species in `specie_ids`. To avoid raising an
         exception when a specie is not in the location map, set `force` to `True`.
 
         Args:
-            species_ids (:obj:`list` of species_ids): a list of species ids
-            force (:obj:`boolean`, optional): if set, do not raise an exception if a species_id in
-                `species_ids` is not found in the species location map.
+            specie_ids (:obj:`list` of specie_ids): a list of species ids
+            force (:obj:`boolean`, optional): if set, do not raise an exception if a specie_id in
+                `specie_ids` is not found in the species location map.
 
         Raises:
-            :obj:`SpeciesPopulationError`: if `force` is False and any species_id in `species_ids` is not in the
+            :obj:`SpeciesPopulationError`: if `force` is False and any specie_id in `specie_ids` is not in the
                 species location map.
         """
         if force:
-            for species_id in species_ids:
+            for specie_id in specie_ids:
                 try:
-                    del self.species_locations[species_id]
+                    del self.species_locations[specie_id]
                 except KeyError:
                     pass
         else:
-            unassigned = list(filter(lambda s: s not in self.species_locations.keys(), species_ids))
+            unassigned = list(filter(lambda s: s not in self.species_locations.keys(), specie_ids))
             if unassigned:
                 raise SpeciesPopulationError("del_species_locations: species {} are not in the location "
-                                             "map.".format(sorted(unassigned)))
-            for species_id in species_ids:
-                del self.species_locations[species_id]
+                    "map.".format(sorted(unassigned)))
+            for specie_id in specie_ids:
+                del self.species_locations[specie_id]
 
-    def locate_species(self, species_ids):
+    def locate_species(self, specie_ids):
         """ Locate the component(s) that store a set of species
 
         Given a list of species identifiers in `species_ids`, partition them into the storage
         component(s) that store their populations. This method is widely used by code that accesses
         species. It returns a dictionary that maps from store name to the ids of species whose
         populations are modeled by the store.
 
         The special name `LOCAL_POP_STORE` represents a special store, the local
-        `wc_sim.local_species_population.LocalSpeciesPopulation` instance. Each other
+        `wc_sim.multialgorithm.local_species_population.LocalSpeciesPopulation` instance. Each other
         store is identified by the name of a remote
-        `from wc_sim.species_pop_sim_object.SpeciesPopSimObject` instance.
+        `from wc_sim.multialgorithm.species_pop_sim_object.SpeciesPopSimObject` instance.
 
         Args:
-            species_ids (:obj:`list` of :obj:`str`): a list of species identifiers
+            specie_ids (:obj:`list` of `str`): a list of species identifiers
 
         Returns:
             dict: a map from store_name -> a set of species_ids whose populations are stored
                 by component store_name.
 
         Raises:
-            :obj:`SpeciesPopulationError`: if a store cannot be found for a species_id in `species_ids`
+            :obj:`SpeciesPopulationError`: if a store cannot be found for a specie_id in `specie_ids`
         """
-        unknown = list(filter(lambda s: s not in self.species_locations.keys(), species_ids))
+        unknown = list(filter(lambda s: s not in self.species_locations.keys(), specie_ids))
         if unknown:
             raise SpeciesPopulationError("locate_species: species {} are not "
-                                         "in the location map.".format(sorted(unknown)))
+                "in the location map.".format(sorted(unknown)))
         inverse_loc_map = defaultdict(set)
-        for species_id in species_ids:
-            store = self.species_locations[species_id]
-            inverse_loc_map[store].add(species_id)
+        for specie_id in specie_ids:
+            store = self.species_locations[specie_id]
+            inverse_loc_map[store].add(specie_id)
         return inverse_loc_map
 
-    def read_one(self, time, species_id):
-        """ Obtain the predicted population of specie `species_id` at the time `time`
+    def read_one(self, time, specie_id):
+        """ Obtain the predicted population of specie `specie_id` at the time `time`
 
         If the specie is stored in the local_pop_store, obtain its population there. Otherwise obtain
         the population from the species_population_cache. If the specie's primary store is a
         remote_pop_store, then its population should be in the cache because the population should
         have been prefetched.
 
         Args:
             time (:obj:`float`): the time at which the population should be obtained
-            species_id (:obj:`str`): identifier of the specie whose population will be obtained.
+            specie_id (:obj:`str`): identifier of the specie whose population will be obtained.
 
         Returns:
-            float: the predicted population of `species_id` at simulation time `time`.
+            float: the predicted population of `specie_id` at simulation time `time`.
 
         Raises:
-            :obj:`SpeciesPopulationError`: if `species_id` is an unknown specie
+            :obj:`SpeciesPopulationError`: if `specie_id` is an unknown specie
         """
-        if species_id not in self.species_locations:
+        if specie_id not in self.species_locations:
             raise SpeciesPopulationError("read_one: specie '{}' not in the location map.".format(
-                species_id))
-        store = self.species_locations[species_id]
-        if store == LOCAL_POP_STORE:
-            return self.local_pop_store.read_one(time, species_id)
+                specie_id))
+        store = self.species_locations[specie_id]
+        if store==LOCAL_POP_STORE:
+            return self.local_pop_store.read_one(time, specie_id)
         else:
             # TODO(Arthur): convert print() to log message
             # print('submodel {} reading {} from cache at {:.2f}'.format(self.submodel.name,
-            #   species_id, time))
-            return self.species_population_cache.read_one(time, species_id)
+            #   specie_id, time))
+            return self.species_population_cache.read_one(time, specie_id)
 
     def read(self, time, species_ids):
         """ Obtain the population of the species identified in `species_ids` at the time `time`
 
         Obtain the species from the local_pop_store and/or the species_population_cache. If some of
         the species' primary stores are remote_pop_stores, then their populations should be in the
         cache because they should have been prefetched.
 
         Args:
             time (:obj:`float`): the time at which the population should be obtained
             species_ids (set): identifiers of the species whose populations will be obtained.
 
         Returns:
             dict: species_id -> population; the predicted population of all requested species at
             time `time`.
 
         Raises:
-            :obj:`SpeciesPopulationError`: if a store cannot be found for a species_id in `species_ids`
+            :obj:`SpeciesPopulationError`: if a store cannot be found for a specie_id in `specie_ids`
             :obj:`SpeciesPopulationError`: if any of the species were cached at a time that differs from `time`
         """
         local_species = self.locate_species(species_ids)[LOCAL_POP_STORE]
         remote_species = set(species_ids) - set(local_species)
 
         local_pops = self.local_pop_store.read(time, local_species)
         cached_pops = self.species_population_cache.read(time, remote_species)
 
         cached_pops.update(local_pops)
         return cached_pops
 
     def adjust_discretely(self, time, adjustments):
         """ A discrete submodel adjusts the population of a set of species at the time `time`
 
         Distribute the adjustments among the population stores managed by this object.
         Iterate through the components that store the population of species listed in `adjustments`.
         Update the local population store immediately and send AdjustPopulationByDiscreteSubmodel
         messages to the remote stores. Since these messages are asynchronous, this method returns
         as soon as they are sent.
 
         Args:
             time (:obj:`float`): the time at which the population is being adjusted
-            adjustments (:obj:`dict` of `float`): map: species_ids -> population_adjustment; adjustments
+            adjustments (:obj:`dict` of `float`): map: specie_ids -> population_adjustment; adjustments
                 to be made to some species populations
 
         Returns:
             :obj:`list`: the names of the stores for the species whose populations are adjusted
         """
-        stores = []
-        for store, species_ids in self.locate_species(adjustments.keys()).items():
+        stores=[]
+        for store,species_ids in self.locate_species(adjustments.keys()).items():
             stores.append(store)
             store_adjustments = DictUtil.filtered_dict(adjustments, species_ids)
-            if store == LOCAL_POP_STORE:
+            if store==LOCAL_POP_STORE:
                 self.local_pop_store.adjust_discretely(time, store_adjustments)
             else:
                 self.submodel.send_event(time-self.submodel.time,
-                                         self.remote_pop_stores[store],
-                                         message_types.AdjustPopulationByDiscreteSubmodel(store_adjustments))
+                    self.remote_pop_stores[store],
+                    message_types.AdjustPopulationByDiscreteSubmodel(store_adjustments))
         return stores
 
     def adjust_continuously(self, time, adjustments):
         """ A continuous submodel adjusts the population of a set of species at the time `time`
 
         Args:
             time (:obj:`float`): the time at which the population is being adjusted
-            adjustments (:obj:`dict` of `tuple`): map: species_ids -> (population_adjustment, flux);
-                adjustments to be made to some species populations.
+            population_slopes (:obj:`dict` of `float`): map: specie_id -> population_slope;
+                updated population slopes for some, or all, species populations
 
         See the description for `adjust_discretely` above.
 
         Returns:
             list: the names of the stores for the species whose populations are adjusted.
         """
-        stores = []
-        for store, species_ids in self.locate_species(adjustments.keys()).items():
+        stores=[]
+        for store,species_ids in self.locate_species(adjustments.keys()).items():
             stores.append(store)
             store_adjustments = DictUtil.filtered_dict(adjustments, species_ids)
-            if store == LOCAL_POP_STORE:
+            if store==LOCAL_POP_STORE:
                 self.local_pop_store.adjust_continuously(time, store_adjustments)
             else:
                 self.submodel.send_event(time-self.submodel.time,
-                                         self.remote_pop_stores[store],
-                                         message_types.AdjustPopulationByContinuousSubmodel(store_adjustments))
+                    self.remote_pop_stores[store],
+                    message_types.AdjustPopulationByContinuousSubmodel(store_adjustments))
         return stores
 
     def prefetch(self, delay, species_ids):
         """ Obtain species populations from remote stores when they will be needed in the future
 
         Generate GetPopulation queries that obtain species populations whose primary stores are
         remote at `delay` in the future. The primary stores (`SpeciesPopSimObject` objects)
         will respond to the GetPopulation queries with GivePopulation responses.
 
         To ensure that the remote store object executes the GetPopulation at an earlier simulation
         time than the submodel will need the data, decrease the event time of the GetPopulation
         event to the previous floating point value.
 
         Args:
             delay (:obj:`float`): the populations will be needed at now + `delay`
-            species_ids (:obj:`list` of species_ids): a list of species ids
+            specie_ids (:obj:`list` of specie_ids): a list of species ids
 
         Returns:
             list: the names of the stores for the species whose populations are adjusted.
         """
         # TODO(Arthur): IMPORTANT optimizations: reduce rate of prefetch
         # 1: store most species locally
         # 2: instead of sending GetPopulation messages to retrieve populations may be unchanged,
         #   make write-through caches which push population updates from reaction executions
         # 3: draw reaction partition boundaries over species (edges) that rarely update
         # TODO(Arthur): IMPORTANT: consider whether a better mechanism is available for ordering
         #   concurrent events than time shifting by miniscule amounts, as done with epsilon here
-        if delay <= 0:
+        if delay<=0:
             raise SpeciesPopulationError("prefetch: {} provided, but delay must "
-                                         "be non-negative.".format(delay))
-        stores = []
+                "be non-negative.".format(delay))
+        stores=[]
         epsilon = config_multialgorithm['epsilon']
-        for store, species_ids in self.locate_species(species_ids).items():
-            if store != LOCAL_POP_STORE:
+        for store,species_ids in self.locate_species(species_ids).items():
+            if store!=LOCAL_POP_STORE:
                 stores.append(store)
                 # advance the receipt of GetPopulation so the SpeciesPopSimObject executes it before
                 # the submodel needs the value
                 self.submodel.send_event(delay - epsilon*0.5,
-                                         self.remote_pop_stores[store],
-                                         message_types.GetPopulation(set(species_ids)))
+                    self.remote_pop_stores[store],
+                    message_types.GetPopulation(set(species_ids)))
         return stores
 
     def __str__(self):
         """ Provide readable AccessSpeciesPopulations state
 
         Provide the submodel's name, the name of the local_pop_store, and the id and store name of
         each specie accessed by this AccessSpeciesPopulations.
 
         Returns:
             :obj:`str`: a multi-line string describing this AccessSpeciesPopulations' state.
         """
-        state = ['AccessSpeciesPopulations state:']
+        state=['AccessSpeciesPopulations state:']
         if hasattr(self, 'submodel'):
             state.append('submodel: {}'.format(self.submodel.id))
         state.append('local_pop_store: {}'.format(self.local_pop_store.name))
         state.append('species locations:')
-        state.append('species_id\tstore_name')
-        state += ['{}\t{}'.format(k, self.species_locations[k])
-                  for k in sorted(self.species_locations.keys())]
+        state.append('specie_id\tstore_name')
+        state += ['{}\t{}'.format(k,self.species_locations[k])
+            for k in sorted(self.species_locations.keys())]
         return '\n'.join(state)
 
 
 # TODO(Arthur): cover after MVP wc_sim done
 epsilon = config_multialgorithm['epsilon']  # pragma: no cover
-
-
 class SpeciesPopulationCache(object):       # pragma: no cover
     """ Cache the population of species whose primary stores are remote population stores
 
     Attributes:
         access_species_populations (:obj:`AccessSpeciesPopulations`): the `AccessSpeciesPopulations`
             containing this Object.
-        _cache (:obj:`dict` of :obj:`tuple`): map: species_id -> (time, population); the species
+        _cache (:obj:`dict` of :obj:`tuple`): map: specie_id -> (time, population); the species
             whose counts are cached, containing the last write time in `time`, and the population.
     """
 
     def __init__(self, access_species_populations):
         """ Initialize a SpeciesPopulationCache object """
         self.access_species_populations = access_species_populations
         self._cache = {}
 
     def clear_cache(self):
         """ Clear the cache """
         self._cache = {}
 
     def cache_population(self, time, populations):
         """ Cache some population values
 
         Args:
             time (:obj:`float`): the time of the cached values
-            populations (:obj:`dict` of float): map: species_ids -> population; the population
+            populations (:obj:`dict` of float): map: specie_ids -> population; the population
                 of the species at `time`.
 
         Raises:
             :obj:`SpeciesPopulationError`: if the species are stored in the ASP's local store; they should
                 not be cached.
             :obj:`SpeciesPopulationError`: if, for any specie, `time` is not greater than its previous
                 cache time.
         """
         # raise exception if the species are stored in the ASP's local store
         store_name_map = self.access_species_populations.locate_species(populations.keys())
         if LOCAL_POP_STORE in store_name_map:
             raise SpeciesPopulationError("cache_population: some species are stored in the "
-                                         "AccessSpeciesPopulations's local store: {}.".format(
-                                             list(store_name_map[LOCAL_POP_STORE])))
+                "AccessSpeciesPopulations's local store: {}.".format(
+                    list(store_name_map[LOCAL_POP_STORE])))
         # TODO(Arthur): could raise an exception if the species are not stored in the ASP's remote stores
-        for species_id, population in populations.items():
+        for specie_id,population in populations.items():
             # raise exception if the time of this cache is not greater than the previous cache time
-            if species_id in self._cache and time <= self._cache[species_id][0]:
+            if specie_id in self._cache and time <= self._cache[specie_id][0]:
                 raise SpeciesPopulationError("cache_population: caching an earlier population: "
-                                             "species_id: {}; current time: {} <= previous time {}.".format(species_id, time,
-                                                                                                            self._cache[species_id][0]))
-            self._cache[species_id] = (time, population)
+                    "specie_id: {}; current time: {} <= previous time {}.".format(specie_id, time,
+                    self._cache[specie_id][0]))
+            self._cache[specie_id] = (time, population)
 
-    def read_one(self, time, species_id):
+    def read_one(self, time, specie_id):
         """ Obtain the cached population of a specie at a particular time
 
         Args:
             time (:obj:`float`): the expected time of the cached values
-            species_id (:obj:`str`): identifier of the specie to obtain.
+            specie_id (:obj:`str`): identifier of the specie to obtain.
 
         Returns:
-            float: the cached population of `species_id` at simulation time `time`.
+            float: the cached population of `specie_id` at simulation time `time`.
 
         Raises:
             :obj:`SpeciesPopulationError`: if the species are stored in the ASP's local store, which means
                 that they should not be cached.
             :obj:`SpeciesPopulationError`: if `time` is not greater than a specie's previous cache time
         """
-        if species_id not in self._cache:
+        if specie_id not in self._cache:
             raise SpeciesPopulationError("SpeciesPopulationCache.read_one: specie '{}' not "
-                                         "in cache.".format(species_id))
-        if self._cache[species_id][0] + epsilon < time:
+                "in cache.".format(specie_id))
+        if self._cache[specie_id][0] + epsilon < time:
             raise SpeciesPopulationError("SpeciesPopulationCache.read_one: cache age of {} too big "
-                                         "for read at time {} of specie '{}'.".format(
-                                             time-self._cache[species_id][0], time, species_id))
-        return self._cache[species_id][1]
+                "for read at time {} of specie '{}'.".format(
+                    time-self._cache[specie_id][0], time, specie_id))
+        return self._cache[specie_id][1]
 
     def read(self, time, species_ids):
         """ Read the cached population of a set of species at a particular time
 
         Args:
             time (:obj:`float`): the time at which the population should be obtained
             species_ids (set): identifiers of the species to read.
 
         Returns:
             species counts: dict: species_id -> copy_number; the cached copy number of each
             requested species at simulation time `time`.
 
         Raises:
             :obj:`SpeciesPopulationError`: if any of the species are not stored in the cache
             :obj:`SpeciesPopulationError`: if any of the species were cached at a time that differs
                 from `time`.
         """
         missing = list(filter(lambda specie: specie not in self._cache, species_ids))
         if missing:
             raise SpeciesPopulationError("SpeciesPopulationCache.read: species {} not in cache.".format(
                 str(missing)))
         mistimed = list(filter(lambda s_id: self._cache[s_id][0] + epsilon < time, species_ids))
         if mistimed:
             raise SpeciesPopulationError("SpeciesPopulationCache.read: species {} not reading "
-                                         "recently cached value(s).".format(str(mistimed)))
-        return {specie: self._cache[specie][1] for specie in species_ids}
+                "recently cached value(s).".format(str(mistimed)))
+        return {specie:self._cache[specie][1] for specie in species_ids}
 
 
 # logging
 debug_log = debug_logs.get_log('wc.debug.file')
 
 # TODO(Arthur): after MVP wc_sim is done, replace references to LocalSpeciesPopulation with
 # references to AccessSpeciesPopulations
-
-
 class LocalSpeciesPopulation(AccessSpeciesPopulationInterface):
     """ Maintain the population of a set of species
 
     `LocalSpeciesPopulation` tracks the population of a set of species. Population values (copy numbers)
     can be read or modified (adjusted). To enable multi-algorithmic modeling, it supports writes to
     a specie's population by both discrete time and continuous time submodels.
 
     All access operations that read or modify the population must provide a simulation time.
     For any given specie, all operations must occur in non-decreasing simulation time order.
-    Record history operations must also occur in time order.
 
     Simulation time arguments enable detection of temporal causality errors by shared accesses from
     different submodels in a sequential
     simulator. In particular, every read operation must access the previous modification.
 
     A `LocalSpeciesPopulation` object is accessed via local method calls. It can be wrapped as a
     DES simulation object -- a `SimulationObject` -- to provide distributed access, as is done by
     `SpeciesPopSimObject`.
 
     Attributes:
-        name (:obj:`str`): the name of this object.
+        name (:obj:`str`): the name of this object
         time (:obj:`float`): the time of the most recent access to this `LocalSpeciesPopulation`
-        _population (:obj:`dict` of :obj:`Specie`): map: species_id -> Specie(); the species whose
-            counts are stored, represented by Specie objects.
-        _molecular_weights (:obj:`dict` of `float`): map: species_id -> molecular_weight; the
+        _population (:obj:`dict` of :obj:`DynamicSpecie`): map: specie_id -> DynamicSpecie(); the species whose
+            counts are stored, represented by DynamicSpecie objects.
+        _molecular_weights (:obj:`dict` of `float`): map: specie_id -> molecular_weight; the
             molecular weight of each specie
         last_access_time (:obj:`dict` of `float`): map: species_name -> last_time; the last time at
             which the specie was accessed.
-        history (:obj:`dict`) nested dict; an optional history of the species' state. The population
-            history is recorded at each continuous adjustment.
         random_state (:obj:`numpy.random.RandomState`): a PRNG used by all `Species`
+        _record_operations (:obj:`bool`): whether a history is retained for all operations on all species
     """
-    # TODO(Arthur): support tracking the population history of species added at any time
-    # in the simulation
-    # TODO(Arthur): report an error if a Specie is updated by multiple continuous submodels
-    # because each of them assumes that they model all changes to its population over their time step
-    # TODO(Arthur): molecular_weights should provide MW of each species type, as that's what the model has
-
-    def __init__(self, name, initial_population, molecular_weights, initial_fluxes=None,
-                 random_state=None, retain_history=True):
+    # TODO(Arthur): report an error if a DynamicSpecie is updated by multiple continuous submodels
+    # TODO(Arthur): molecular_weights should accept MW of each species type, like the model does
+    # TODO(Arthur): specify record_operations and model_continuously independently for each specie
+    def __init__(self, name, initial_population, molecular_weights,
+        initial_time=0, model_continuously=False, record_operations=False):
         """ Initialize a `LocalSpeciesPopulation` object
 
-        Initialize a `LocalSpeciesPopulation` object. Establish its initial population, and initialize
-            the history if `retain_history` is `True`.
-
         Args:
+            name (:obj:`str`): the name of this object
             initial_population (:obj:`dict` of `float`): initial population for some species;
-                dict: species_id -> initial_population
-            molecular_weights (:obj:`dict` of `float`): map: species_id -> molecular_weight,
+                dict: specie_id -> initial_population
+            molecular_weights (:obj:`dict` of `float`): map: specie_id -> molecular_weight,
                 provided for computing the mass of lists of species in a `LocalSpeciesPopulation`
-            initial_fluxes (:obj:`dict` of `float`, optional): map: species_id -> initial_flux;
-                initial fluxes for all species whose populations are estimated by a continuous
-                submodel. Fluxes are ignored for species not specified in initial_population.
-            retain_history (:obj:`bool`, optional): whether to retain species population history
+            initial_time (:obj:`float`, optional): the initialization time; defaults to 0
+            model_continuously (:obj:`bool`, optional): whether to each species is modeled continuously
+            record_operations (:obj:`bool`, optional): whether to retain a history of all operations on all species
 
         Raises:
             :obj:`SpeciesPopulationError`: if the population cannot be initialized
         """
         self.name = name
-        self.time = 0
+        self.time = initial_time
         self._population = {}
         self.last_access_time = {}
-        self.random_state = random_state
-
-        if retain_history:
-            self._initialize_history()
-
-        for species_id in initial_population:
-            if initial_fluxes is not None and species_id in initial_fluxes:
-                self.init_cell_state_specie(species_id, initial_population[species_id],
-                                            initial_fluxes[species_id])
-            else:
-                self.init_cell_state_specie(species_id, initial_population[species_id])
+        self.random_state = RandomStateManager.instance()
 
         unknown_weights = set(initial_population.keys()) - set(molecular_weights.keys())
         if unknown_weights:
             # raise exception if any species are missing weights
             raise SpeciesPopulationError("Cannot init LocalSpeciesPopulation because some species "
-                                         "are missing weights: {}".format(
-                                             ', '.join(map(lambda x: "'{}'".format(str(x)), unknown_weights))))
+                "are missing weights: {}".format(
+                ', '.join(map(lambda x: "'{}'".format(str(x)), unknown_weights))))
+
+        for specie_id in initial_population:
+            self.init_cell_state_specie(specie_id, initial_population[specie_id], model_continuously,
+                record_operations)
         self._molecular_weights = molecular_weights
+        self._record_operations = record_operations
 
         # log initialization data
         debug_log.debug("initial_population: {}".format(DictUtil.to_string_sorted_by_key(
             initial_population)), sim_time=self.time)
-        debug_log.debug("initial_fluxes: {}".format(DictUtil.to_string_sorted_by_key(initial_fluxes)),
-                        sim_time=self.time)
 
-    def init_cell_state_specie(self, species_id, population, initial_flux_given=None):
-        """ Initialize a specie with the given population and flux
+    def init_cell_state_specie(self, specie_id, population, model_continuously, record_operations=False):
+        """ Initialize a specie with the given population and, optionally, initial population slope
 
         Add a specie to the cell state. The specie's population is set at the current time.
 
         Args:
-            species_id (:obj:`str`): the specie's globally unique identifier
+            specie_id (:obj:`str`): the specie's globally unique identifier
             population (:obj:`float`): initial population of the specie
-            initial_flux_given (:obj:`float`, optional): an initial flux for the specie
+            model_continuously (:obj:`bool`): whether the species is modeled continuously
+            record_operations (:obj:`bool`, optional): whether to retain a history of all operations
+                on the specie; default = `False`
 
         Raises:
             :obj:`SpeciesPopulationError`: if the specie is already stored by this LocalSpeciesPopulation
         """
-        if species_id in self._population:
-            raise SpeciesPopulationError("species_id '{}' already stored by this "
-                                         "LocalSpeciesPopulation".format(species_id))
-        self._population[species_id] = Specie(species_id, self.random_state, population,
-                                              initial_flux=initial_flux_given)
-        self.last_access_time[species_id] = self.time
-        self._add_to_history(species_id)
+        if specie_id in self._population:
+            raise SpeciesPopulationError("specie_id '{}' already stored by this "
+                "LocalSpeciesPopulation".format(specie_id))
+        self._population[specie_id] = DynamicSpecie(specie_id, self.random_state, population,
+            modeled_continuously=model_continuously, record_history=record_operations)
+        self.last_access_time[specie_id] = self.time
 
     def _all_species(self):
         """ Return the IDs species known by this LocalSpeciesPopulation
 
         Returns:
             :obj:`set`: the species known by this LocalSpeciesPopulation
         """
         return set(self._population.keys())
 
     def _check_species(self, time, species=None):
         """ Check whether the species are a set, or not known by this LocalSpeciesPopulation
 
         Also checks whether the species are being accessed in time order.
 
         Args:
             time (:obj:`float`): the time at which the population might be accessed
             species (:obj:`set`, optional): set of species_ids; if not supplied, read all species
 
         Raises:
             :obj:`SpeciesPopulationError`: if species is not a set
             :obj:`SpeciesPopulationError`: if any species in `species` do not exist
             :obj:`SpeciesPopulationError`: if a specie in `species` is being accessed at a time earlier
                 than a prior access.
         """
         if not species is None:
             if not isinstance(species, set):
                 raise SpeciesPopulationError("species '{}' must be a set".format(species))
             unknown_species = species - self._all_species()
             if unknown_species:
                 # raise exception if some species are non-existent
                 raise SpeciesPopulationError("request for population of unknown specie(s): {}".format(
                     ', '.join(map(lambda x: "'{}'".format(str(x)), unknown_species))))
             early_accesses = list(filter(lambda s: time < self.last_access_time[s], species))
             if early_accesses:
                 raise SpeciesPopulationError("access at time {} is an earlier access of specie(s) {} than at {}".format(
                     time, early_accesses, [self.last_access_time[s] for s in early_accesses]))
 
     def _update_access_times(self, time, species=None):
         """ Update the access time to `time` for all species_ids in `species`
 
         Args:
             time (:obj:`float`): the access time which should be set for the species
             species (:obj:`set`, optional): a set of species_ids; if not provided, read all species
         """
         if species is None:
             species = self._all_species()
-        for species_id in species:
-            self.last_access_time[species_id] = time
+        for specie_id in species:
+            self.last_access_time[specie_id] = time
 
-    def read_one(self, time, species_id):
-        """ Obtain the predicted population of specie `species_id` at simulation time `time`
+    def read_one(self, time, specie_id):
+        """ Obtain the predicted population of specie `specie_id` at simulation time `time`
 
         Args:
             time (:obj:`float`): the time at which the population should be estimated
-            species_id (:obj:`str`): identifier of the specie to access.
+            specie_id (:obj:`str`): identifier of the specie to access.
 
         Returns:
-            float: the predicted population of `species_id` at simulation time `time`.
+            :obj:`float`: the predicted population of `specie_id` at simulation time `time`.
 
         Raises:
             :obj:`SpeciesPopulationError`: if the population of an unknown specie was requested
         """
-        species_id_in_set = {species_id}
-        self._check_species(time, species_id_in_set)
+        specie_id_in_set = {specie_id}
+        self._check_species(time, specie_id_in_set)
         self.time = time
-        self._update_access_times(time, species_id_in_set)
-        return self._population[species_id].get_population(time)
+        self._update_access_times(time, specie_id_in_set)
+        return self._population[specie_id].get_population(time)
 
-    def read(self, time, species=None):
+    def read(self, time, species=None, round=True):
         """ Read the predicted population of a list of species at simulation time `time`
 
         Args:
             time (:obj:`float`): the time at which the population should be estimated
             species (:obj:`set`, optional): identifiers of the species to read; if not supplied, read all species
+            round (:obj:`bool`, optional): if `round` then round the populations to integers
 
         Returns:
-            species counts: dict: species_id -> copy_number; the predicted copy number of each
+            :obj:`dict`: species counts: species_id -> copy_number; the predicted copy number of each
             requested species at `time`
 
         Raises:
             :obj:`SpeciesPopulationError`: if the population of unknown specie(s) are requested
         """
         if species is None:
             species = self._all_species()
         self._check_species(time, species)
         self.time = time
         self._update_access_times(time, species)
-        return {specie: self._population[specie].get_population(time) for specie in species}
+        return {specie:self._population[specie].get_population(time, round=round) for specie in species}
 
     def adjust_discretely(self, time, adjustments):
         """ A discrete submodel adjusts the population of a set of species at simulation time `time`
 
         Args:
             time (:obj:`float`): the simulation time of the population adjustedment
-            adjustments (:obj:`dict` of `float`): map: species_ids -> population_adjustment; adjustments
+            adjustments (:obj:`dict` of `float`): map: specie_ids -> population_adjustment; adjustments
                 to be made to the population of some species
 
         Raises:
             :obj:`SpeciesPopulationError`: if any adjustment attempts to change the population of an
                 unknown species
             :obj:`SpeciesPopulationError`: if any population estimate would become negative
         """
         self._check_species(time, set(adjustments.keys()))
         self.time = time
         errors = []
         for specie in adjustments:
             try:
-                self._population[specie].discrete_adjustment(adjustments[specie], self.time)
+                self._population[specie].discrete_adjustment(self.time, adjustments[specie])
                 self._update_access_times(time, {specie})
                 self.log_event('discrete_adjustment', self._population[specie])
             except NegativePopulationError as e:
                 errors.append(str(e))
         if errors:
             raise SpeciesPopulationError("adjust_discretely error(s) at time {}:\n{}".format(
                 time, '\n'.join(errors)))
 
-    def adjust_continuously(self, time, adjustments):
-        """ A continuous submodel adjusts the population of a set of species at simulation time `time`
+    def adjust_continuously(self, time, population_slopes):
+        """ A continuous submodel adjusts the population slopes of a set of species at simulation time `time`
+
+        Species retain the population slopes to interpolate the population until the next
+        call to `adjust_continuously`.
 
         Args:
             time (:obj:`float`): the time at which the population is being adjusted
-            adjustments (:obj:`dict` of `tuple`): map: species_ids -> (population_adjustment, flux);
-                adjustments to be made to some species populations.
+            population_slopes (:obj:`dict` of `float`): map: specie_id -> population_slope;
+                updated population slopes for some, or all, species populations
 
         Raises:
-            :obj:`SpeciesPopulationError`: if any adjustment attempts to change the population of an
-                unknown species,
-                or if any population estimate would become negative
+            :obj:`SpeciesPopulationError`: if any adjustment attempts to change the population slope of an
+                unknown species.
+            :obj:`SpeciesPopulationError`: if any population estimate would become negative
         """
-        self._check_species(time, set(adjustments.keys()))
+        self._check_species(time, set(population_slopes.keys()))
         self.time = time
 
-        # record simulation state history
-        # TODO(Arthur): maybe also do it in adjust_discretely(); better, separately control its periodicity
-        if self._recording_history():
-            self._record_history()
         errors = []
-        for specie, (adjustment, flux) in adjustments.items():
+        for specie_id, population_slope in population_slopes.items():
             try:
-                self._population[specie].continuous_adjustment(adjustment, time, flux)
-                self._update_access_times(time, [specie])
-                self.log_event('continuous_adjustment', self._population[specie])
+                self._population[specie_id].continuous_adjustment(time, population_slope)
+                self._update_access_times(time, [specie_id])
+                self.log_event('continuous_adjustment', self._population[specie_id])
             except (SpeciesPopulationError, NegativePopulationError) as e:
                 errors.append(str(e))
         if errors:
             # TODO(Arthur): consistently use debug_log
-            debug_log.error("Error: on specie {}: {}".format(specie, '\n'.join(errors)), sim_time=self.time)
+            debug_log.error("Error: on specie {}: {}".format(specie_id, '\n'.join(errors)), sim_time=self.time)
             raise SpeciesPopulationError("adjust_continuously error(s) at time {}:\n{}".format(
                 time, '\n'.join(errors)))
 
-    # TODO(Arthur): probably don't need compartment_id, because compartment is part of the species_ids
-    def compartmental_mass(self, compartment_id, species_ids=None, time=None):
+    def compartmental_mass(self, compartment_id, species_ids=None):
         """ Compute the current mass of some, or all, species in a compartment
 
         Args:
             compartment_id (:obj:`str`): the ID of the compartment
-            species_ids (:obj:`list` of :obj:`str`, optional): identifiers of the species whose mass will be obtained;
+            species_ids (:obj:`list` of `str`, optional): identifiers of the species whose mass will be obtained;
                 if not provided, then compute the mass of all species in the compartment
-            time (number, optional): the current simulation time
 
         Returns:
             :obj:`float`: the current total mass of the specified species in compartment `compartment_id`, in grams
 
         Raises:
-            :obj:`SpeciesPopulationError`: if a specie's molecular weight was not provided to
-                `__init__()` in `molecular_weights`
+            :obj:`SpeciesPopulationError`: if a specie's molecular weight is unavailable
         """
         if species_ids is None:
             species_ids = self._all_species()
-        if time is None:
-            time = self.time
         mass = 0.
-        for species_id in species_ids:
-            _, comp = wc_lang.Species.parse_id(species_id)
+        for specie_id in species_ids:
+            _, comp = get_species_and_compartment_from_name(specie_id)
             if comp == compartment_id:
                 try:
-                    mw = self._molecular_weights[species_id]              
+                    mass += self._molecular_weights[specie_id] * self.read_one(self.time, specie_id)
                 except KeyError as e:
                     raise SpeciesPopulationError("molecular weight not available for '{}'".format(
-                        species_id))
-                if not numpy.isnan(mw):
-                    mass += mw * self.read_one(time, species_id)
-        return mass / Avogadro
+                        specie_id))
+        return mass/Avogadro
+
+    def invalid_weights(self, species_ids=None):
+        """ Find the species that do not have a positive, numerical molecular weight
+
+        Args:
+            species_ids (:obj:`list` of `str`, optional): identifiers of the species whose molecular weights
+                will be checked; if not provided, then check all species
+
+        Returns:
+            :obj:`set`: the ids of species that do not have a positive, numerical molecular weight
+        """
+        if species_ids is None:
+            species_ids = self._all_species()
+        species_with_invalid_mw = set()
+        for specie_id in species_ids:
+            try:
+                mw = self._molecular_weights[specie_id]
+            except KeyError:
+                species_with_invalid_mw.add(specie_id)
+                continue
+            try:
+                if 0 < mw:
+                    continue
+                species_with_invalid_mw.add(specie_id)
+            except (TypeError, ValueError):
+                species_with_invalid_mw.add(specie_id)
+        return species_with_invalid_mw
 
     def log_event(self, message, specie):
         """ Log an event that modifies a specie's population
 
-        Log the event's simulation time, event type, specie population, and current flux (if
+        Log the event's simulation time, event type, specie population, and current population slope (if
         specified).
 
         Args:
             message (:obj:`str`): description of the event's type.
-            specie (:obj:`Specie`): the object whose adjustment is being logged
+            specie (:obj:`DynamicSpecie`): the object whose adjustment is being logged
         """
         try:
-            flux = specie.continuous_flux
+            population_slope = specie.population_slope
         except AttributeError:
-            flux = None
-        values = [message, specie.last_population, flux]
+            population_slope = None
+        values = [message, specie.last_population, population_slope]
         values = map(lambda x: str(x), values)
-        # log Sim_time Adjustment_type New_population New_flux
+        # log Sim_time Adjustment_type New_population New_population_slope
         debug_log.debug('\t'.join(values), local_call_depth=1, sim_time=self.time)
 
-    def _initialize_history(self):
-        """ Initialize the population history with current population """
-        self._history = {}
-        self._history['time'] = [self.time]  # a list of times at which population is recorded
-        # the value of self._history['population'][species_id] is a list of
-        # the population of species_id at the times history is recorded
-        self._history['population'] = {}
-
-    def _add_to_history(self, species_id):
-        """ Add a specie to the history
-
-        Args:
-            species_id (:obj:`str`): a unique specie identifier.
-        """
-        if self._recording_history():
-            population = self.read_one(self.time, species_id)
-            self._history['population'][species_id] = [population]
-
-    def _recording_history(self):
-        """ Is history being recorded?
-
-        Returns:
-            True if history is being recorded.
-        """
-        return hasattr(self, '_history')
-
-    def _record_history(self):
-        """ Record the current population in the history
-
-        Snapshot the current population of all species in the history. The current time
-        is obtained from `self.time`.
-
-        Raises:
-            :obj:`SpeciesPopulationError`: if the current time is not greater than the previous time at which the
-            history was recorded.
-        """
-        # todo: more comprehensible error message here
-        if not self._history['time'][-1] < self.time:
-            raise SpeciesPopulationError("time of previous _record_history() ({}) not less than current time ({})".format(
-                self._history['time'][-1], self.time))
-        self._history['time'].append(self.time)
-        for species_id, population in self.read(self.time, self._all_species()).items():
-            self._history['population'][species_id].append(population)
-
-    # TODO(Arthur): fix this docstring
-    def report_history(self, numpy_format=False, species_type_ids=None, compartment_ids=None):
-        """ Provide the time and species count history
+    def report_history(self, specie_ids=None):
+        """ Provide the history of operations on species in this LocalSpeciesPopulation
 
         Args:
-            numpy_format (:obj:`bool`, optional): if set, return history in a 3 dimensional numpy array
-            species_type_ids (:obj:`list` of :obj:`str`, optional): the ids of species_types in the
-                `Model` being simulated
-            compartment_ids (:obj:`list` of :obj:`str`, optional): the ids of the compartments in the
-                `Model` being simulated
+            specie_ids (:obj:`iterator` of `str`, optional): the ids of species to provide;
+                if `None`provide the history of operations on all species
 
         Returns:
-            :obj:`dict`: The time and species count history. By default, return a `dict`, with
-            `rv['time']` = list of time samples
-            `rv['population'][species_id]` = list of counts for species_id at the time samples
-            If `numpy_format` set, return a tuple containing a pair of numpy arrays that contain
-            the time and population histories, respectively.
+            :obj:`dict` of `list`: a map from species_id to history for each specie
 
         Raises:
             :obj:`SpeciesPopulationError`: if the history was not recorded
-            :obj:`SpeciesPopulationError`: if `numpy_format` set but `species` or `compartments` are
-                not provided
-        """
-        if not self._recording_history():
-            raise SpeciesPopulationError("history not recorded")
-        if numpy_format:
-            if species_type_ids is None or compartment_ids is None:
-                raise SpeciesPopulationError(
-                    "species_type_ids and compartment_ids must be provided if numpy_format is set")
-            time_hist = numpy.asarray(self._history['time'])
-            species_counts_hist = numpy.zeros((len(species_type_ids), len(compartment_ids),
-                                               len(self._history['time'])))
-            for species_type_index, species_type_id in list(enumerate(species_type_ids)):
-                for comp_index, compartment_id in list(enumerate(compartment_ids)):
-                    for time_index in range(len(self._history['time'])):
-                        species_id = wc_lang.Species._gen_id(species_type_id, compartment_id)
-                        if species_id in self._history['population']:
-                            species_counts_hist[species_type_index, comp_index, time_index] = \
-                                self._history['population'][species_id][time_index]
-
-            return (time_hist, species_counts_hist)
-        else:
-            return self._history
-
-    def history_debug(self):
-        """ Provide some of the history in a string
-
-        Provide a string containing the start and end time of the history and
-        a table with the first and last population value for each specie.
-
-        Returns:
-            :obj:`str`: the start and end time of he history and a
-                tab-separated matrix of rows with species id, first, last population values
-
-        Raises:
-            :obj:`SpeciesPopulationError`: if the history was not recorded.
         """
-        if self._recording_history():
-            lines = []
-            lines.append("#times\tfirst\tlast")
-            lines.append("{}\t{}\t{}".format(len(self._history['time']), self._history['time'][0],
-                                             self._history['time'][-1]))
-            lines.append("Specie\t#values\tfirst\tlast")
-            for s in self._history['population'].keys():
-                lines.append("{}\t{}\t{:.1f}\t{:.1f}".format(s, len(self._history['population'][s]),
-                                                             self._history['population'][s][0], self._history['population'][s][-1]))
-            return '\n'.join(lines)
-        else:
-            raise SpeciesPopulationError("history not recorded")
+        if not self._record_operations:
+            raise SpeciesPopulationError("history was not recorded")
+        if specie_ids is None:
+            specie_ids = self._population.keys()
+        history = {id: self._population[id].get_history() for id in specie_ids}
+        return history
 
     def __str__(self):
         """ Provide readable `LocalSpeciesPopulation` state
 
         Provide the name of this `LocalSpeciesPopulation`, the current time, and the id, population
         of each specie stored by this object. Species modeled by continuous time submodels also
-        have the most recent continuous time adjustment and the current flux.
+        have the most recent continuous time adjustment and the current population slope.
 
         Returns:
             :obj:`str`: a multi-line string describing this LocalSpeciesPopulation's state
         """
-        state = []
+        state=[]
         state.append('name: {}'.format(self.name))
         state.append('time: {}'.format(str(self.time)))
-        state.append(Specie.heading())
-        for species_id in sorted(self._population.keys()):
-            state.append(self._population[species_id].row())
+        state.append(DynamicSpecie.heading())
+        for specie_id in sorted(self._population.keys()):
+            state.append(self._population[specie_id].row())
         return '\n'.join(state)
 
 
 class MakeTestLSP(object):
     """ Make a LocalSpeciesPopulation for testing
 
     Because a LocalSpeciesPopulation takes about 10 lines of code to make, and they're
     widely needed for testing wc_sim, provide a configurable class that creates a test LSP.
 
     Attributes:
         local_species_pop (:obj:`LocalSpeciesPopulation`): the `LocalSpeciesPopulation` created
     """
     DEFAULT_NUM_SPECIES = 10
     DEFAULT_ALL_POPS = 1E6
     DEFAULT_ALL_MOL_WEIGHTS = 50
-
-    def __init__(self, name=None, initial_population=None, molecular_weights=None, initial_fluxes=None,
-                 retain_history=True, **kwargs):
+    def __init__(self, name=None, initial_population=None, molecular_weights=None, initial_population_slopes=None,
+        record_operations=False, **kwargs):
         """ Initialize a `MakeTestLSP` object
 
         All initialized arguments are applied to the local species population being created.
         Valid keys in `kwargs` are `num_species`, `all_pops`, and `all_mol_weights`, which default to
         `MakeTestLSP.DEFAULT_NUM_SPECIES`, `MakeTestLSP.DEFAULT_ALL_POPS`, and
         `MakeTestLSP.DEFAULT_ALL_MOL_WEIGHTS`, respectively. These make a uniform population of
         num_species, with population of all_pops, and molecular weights of all_mol_weights
 
         Args:
             name (:obj:`str`, optional): the name of the local species population being created
             initial_population (:obj:`dict` of `float`, optional): initial population for some species;
-                dict: species_id -> initial_population
-            molecular_weights (:obj:`dict` of `float`, optional): map: species_id -> molecular_weight,
+                dict: specie_id -> initial_population
+            molecular_weights (:obj:`dict` of `float`, optional): map: specie_id -> molecular_weight,
                 provided for computing the mass of lists of species in a `LocalSpeciesPopulation`
-            initial_fluxes (:obj:`dict` of `float`, optional): map: species_id -> initial_flux;
-                initial fluxes for all species whose populations are estimated by a continuous
-                submodel. Fluxes are ignored for species not specified in initial_population.
-            retain_history (:obj:`bool`, optional): whether to retain species population history
+            initial_population_slopes (:obj:`dict` of `float`, optional): map: specie_id -> initial_population_slope;
+                initial population slopes for all species whose populations are estimated by a continuous
+                submodel. Population slopes are ignored for species not specified in initial_population.
+            record_operations (:obj:`bool`, optional): whether to record the history of operations
         """
         name = 'test_lsp' if name is None else name
         if initial_population is None:
             self.num_species = kwargs['num_species'] if 'num_species' in kwargs else MakeTestLSP.DEFAULT_NUM_SPECIES
             self.species_nums = list(range(0, self.num_species))
             self.all_pops = kwargs['all_pops'] if 'all_pops' in kwargs else MakeTestLSP.DEFAULT_ALL_POPS
             comp_id = 'comp_id'
-            self.species_ids = list(map(lambda x: "species_{}[{}]".format(x, comp_id), self.species_nums))
+            self.species_ids = list(map(lambda x: "specie_{}[{}]".format(x, comp_id), self.species_nums))
             self.initial_population = dict(zip(self.species_ids, [self.all_pops]*len(self.species_nums)))
         else:
             self.initial_population = initial_population
             self.species_ids = list(initial_population.keys())
 
         if molecular_weights is None:
             if 'all_mol_weights' in kwargs:
                 self.all_mol_weights = kwargs['all_mol_weights']
             else:
                 self.all_mol_weights = MakeTestLSP.DEFAULT_ALL_MOL_WEIGHTS
             self.molecular_weights = dict(zip(self.species_ids, [self.all_mol_weights]*len(self.species_ids)))
         else:
             self.molecular_weights = molecular_weights
-        random_state = RandomStateManager.instance()
-        self.local_species_pop = LocalSpeciesPopulation(name, self.initial_population, self.molecular_weights,
-                                                        initial_fluxes=initial_fluxes, random_state=random_state, retain_history=initial_fluxes)
+        self.local_species_pop = LocalSpeciesPopulation(name, self.initial_population, self.molecular_weights)
+        # todo: use initial_population_slopes and record_operations
 
 
 # TODO(Arthur): cover after MVP wc_sim done
 class SpeciesPopSimObject(LocalSpeciesPopulation, ApplicationSimulationObject,
-                          metaclass=AppSimObjAndABCMeta):  # pragma: no cover
+    metaclass=AppSimObjAndABCMeta): # pragma: no cover
     """ Maintain the population of a set of species in a simulation object that can be parallelized
 
     A whole-cell PDES must run multiple submodels in parallel. These share cell state, such as
     species populations, by accessing shared simulation objects. A SpeciesPopSimObject provides that
     functionality by wrapping a LocalSpeciesPopulation in a `SimulationObject`.
     """
 
     def send_initial_events(self): pass
     """ No initial events to send"""
 
     def get_state(self):
         return 'object state to be provided'
 
-    def __init__(self, name, initial_population, molecular_weights, initial_fluxes=None,
-                 random_state=None, retain_history=True):
+    def __init__(self, name, initial_population, molecular_weights, initial_population_slopes=None):
         """ Initialize a SpeciesPopSimObject object
 
         Initialize a SpeciesPopSimObject object. Initialize its base classes.
 
         Args:
             name (:obj:`str`): the name of the simulation object and local species population object.
 
         For remaining args and exceptions, see `__init__()` documentation for
-        `de_sim.simulation_object.SimulationObject` and `wc_sim.LocalSpeciesPopulation`.
+        `wc_sim.multialgorithm.SimulationObject` and `wc_sim.multialgorithm.LocalSpeciesPopulation`.
+        (Perhaps Sphinx can automate this, but the documentation is unclear.)
         """
         SimulationObject.__init__(self, name)
         LocalSpeciesPopulation.__init__(self, name, initial_population, molecular_weights,
-                                        initial_fluxes, random_state=random_state)
+            initial_population_slopes)
 
     def handle_adjust_discretely_event(self, event):
         """ Handle a simulation event
 
         Args:
-            event (:obj:`de_sim.event.Event`): an `Event` to process
+            event (:obj:`wc_sim.core.Event`): an `Event` to process
         """
         population_change = event.message.population_change
         self.adjust_discretely(self.time, population_change)
 
     def handle_adjust_continuously_event(self, event):
         """ Handle a simulation event
 
         Args:
-            event (:obj:`de_sim.event.Event`): an `Event` to process
+            event (:obj:`wc_sim.core.Event`): an `Event` to process
 
         Raises:
             :obj:`SpeciesPopulationError`: if an `AdjustPopulationByContinuousSubmodel` event acts on a
                 non-existent species.
         """
         population_change = event.message.population_change
         self.adjust_continuously(self.time, population_change)
 
     def handle_get_population_event(self, event):
         """ Handle a simulation event
 
         Args:
-            event (:obj:`de_sim.event.Event`): an `Event` to process
+            event (:obj:`wc_sim.core.Event`): an `Event` to process
 
         Raises:
             :obj:`SpeciesPopulationError`: if a `GetPopulation` message requests the population of an
                 unknown species.
         """
         species = event.message.species
         self.send_event(0, event.sending_object,
-                        message_types.GivePopulation(self.read(self.time, species)))
+            message_types.GivePopulation(self.read(self.time, species)))
 
     def handle_get_current_property_event(self, event):
         """ Handle a simulation event
 
         Args:
-            event (:obj:`de_sim.event.Event`): an `Event` to process
+            event (:obj:`wc_sim.core.Event`): an `Event` to process
 
         Raises:
             :obj:`SpeciesPopulationError`: if an `GetCurrentProperty` message requests an unknown
                 property.
         """
         property_name = event.message.property_name
         if property_name == distributed_properties.MASS:
             self.send_event(0, event.sending_object,
-                            message_types.GiveProperty(property_name, self.time, self.mass()))
+                message_types.GiveProperty(property_name, self.time, self.mass()))
         else:
             raise SpeciesPopulationError("Error: unknown property_name: '{}'".format(
                 property_name))
 
     # register the event handler for each type of message received
-    event_handlers = [
-        # At any time instant, messages are processed in this order
-        (message_types.AdjustPopulationByDiscreteSubmodel, handle_adjust_discretely_event),
-        (message_types.AdjustPopulationByContinuousSubmodel, handle_adjust_continuously_event),
-        (message_types.GetPopulation, handle_get_population_event),
-        (message_types.GetCurrentProperty, handle_get_current_property_event)]
+    event_handlers =[
+            # At any time instant, messages are processed in this order
+            (message_types.AdjustPopulationByDiscreteSubmodel, handle_adjust_discretely_event),
+            (message_types.AdjustPopulationByContinuousSubmodel, handle_adjust_continuously_event),
+            (message_types.GetPopulation, handle_get_population_event),
+            (message_types.GetCurrentProperty, handle_get_current_property_event)]
 
     # register the message types sent
     messages_sent = [message_types.GivePopulation, message_types.GiveProperty]
 
 
-# TODO(Arthur): rename to DynamicSpecies and move to dynamic_components
-class Specie(object):
-    """ Specie tracks the population of a single specie in a multi-algorithmic model
+class DynamicSpecie(object):
+    """ Track the population of a single specie in a multi-algorithmic model
 
     A specie is a shared object that can be read and written by multiple submodels in a
     multi-algorithmic model. We assume that a sequence of accesses of a specie instance will
     occur in non-decreasing simulation time order. (This assumption holds for conservative discrete
     event simulations and all committed parts of optimistic parallel simulations like Time Warp.)
 
     Consider a multi-algorithmic model that contains both discrete-time submodels, like the
     stochastic simulation algorithm (SSA), and continuous-time submodels, like ODEs and FBA.
     Discrete-time algorithms change system state at discrete time instants. Continuous-time
     algorithms employ continuous models of state change, and sample these models at time instants
     determined by the algorithm. At these instants, continuous-time models typically
     estimate a specie's population and the population's rate of change. We assume this behavior.
 
-    A specie's state in a multi-algorithmic model may be modeled by multiple submodels which model
+    A specie's state in a multi-algorithmic model may be modeled by multiple submodels that model
     reactions in which the specie participates. These can be multiple discrete-time submodels and
     at most one continuous-time submodel. (If multiple continuous-time submodels were allowed to
     predict reactions that involve a specie, a mechanism would be needed to reconsile conflicting
-    flux values. We have not addressed that issue yet.)
+    `population_slope` values. We have not addressed that issue yet.)
 
     Discrete-time and continuous-time models adjust the state of a species by the methods
     `discrete_adjustment()` and `continuous_adjustment()`, respectively. These adjustments take the
     following forms,
 
-    * `discrete_adjustment(population_change, time)`
-    * `continuous_adjustment(population_change, time, flux)`
+    * `discrete_adjustment(time, population_change)`
+    * `continuous_adjustment(time, population_slope)`
 
     where `population_change` is the increase or decrease in the specie's population, `time` is the
-    time at which that change takes place, and `flux` is the predicted future rate of change of the
+    time at which that change takes place, and `population_slope` is the predicted future rate of change of the
     population.
 
     To improve the accuracy of multi-algorithmic models, we support linear *interpolation* of
     population predictions for species modeled by a continuous-time submodel. An interpolated
-    prediction is based on the most recent continuous-time flux prediction. Thus, we assume
+    prediction is based on the most recent continuous-time population slope prediction. Thus, we assume
     that a population modeled by a continuous model is adjusted sufficiently frequently
-    that the most recent adjustment accurately estimates flux.
+    that the most recent adjustment accurately estimates population slope.
 
     A specie instance stores the most recent value of the specie's population in `last_population`,
     which is initialized when the instance is created. If a specie is modeled by a
-    continuous-time submodel, it also stores the specie's flux in `continuous_flux` and the time
-    of the most recent `continuous_adjustment` in `continuous_time`. Otherwise, `continuous_time`
-    is `None`. Interpolation determines the population prediction `p` at time `t` as::
+    continuous-time submodel, it also stores the specie's rate of change in `population_slope` and the time
+    of the most recent `continuous_adjustment` in `continuous_time`. Interpolation determines the
+    population prediction `p` at time `t` as::
 
         interpolation = 0
-        if not continuous_time is None:
-            interpolation = (t - continuous_time)*continuous_flux
+        if modeled_continuously:
+            interpolation = (t - continuous_time)*population_slope
         p = last_population + interpolation
 
     This approach is completely general, and can be applied to any simulation value
     whose dynamics are predicted by a multi-algorithmic model.
 
     Population values returned by specie's methods use stochastic rounding to provide integer
     values and avoid systematic rounding bias. See more detail in `get_population`'s docstring.
 
     Attributes:
-        species_name (:obj:`str`): the specie's name; not logically needed, but helpful for error
+        specie_name (:obj:`str`): the specie's name; not logically needed, but helpful for error
             reporting, logging, debugging, etc.
         random_state (:obj:`numpy.random.RandomState`): a shared PRNG
         last_population (:obj:`float`): population after the most recent adjustment
-        continuous_submodel (bool): whether one of the submodels modeling the species is a
+        modeled_continuously (:obj:`bool`): whether one of the submodels modeling the species is a
             continuous submodel; must be set at initialization
-        continuous_flux (:obj:`float`): if a continuous submodel is modeling the specie, the flux provided
-            at initialization or by the most recent adjustment by a continuous model
-        continuous_time (:obj:`float`): if a continuous submodel is modeling the specie, the simulation
-            time of initialization (0) or the most recent adjustment by the continuous model
+        population_slope (:obj:`float`): if a continuous submodel is modeling the specie, the rate of
+            change to the population provided at initialization or by the most recent adjustment by a
+            continuous model
+        continuous_time (:obj:`float`): if a continuous submodel is modeling the specie, the most
+            recent adjustment by the continuous model; initialized to None to indicate that a
+            continuous adjustment has not been made yet
+        last_adjustment_time (:obj:`float`): the time of the latest adjustment; used to prevent
+            reads in the past
+        last_read_time (:obj:`float`): the time of the latest read; used to prevent prior adjustments
+        _record_history (:obj:`bool`): whether to record history of operations
+        _history (:obj:`bool`): history of operations
     """
     # use __slots__ to save space
-    __slots__ = ['species_name', 'last_population', 'continuous_submodel', 'continuous_flux', 'continuous_time',
-                 'random_state']
+    __slots__ = ['specie_name', 'random_state', 'last_population', 'modeled_continuously', 'population_slope',
+        'continuous_time', 'last_adjustment_time', 'last_read_time', '_record_history', '_history']
 
-    def __init__(self, species_name, random_state, initial_population, initial_flux=None):
-        """ Initialize a specie object at simulation time 0
+    def __init__(self, specie_name, random_state, initial_population, modeled_continuously=False,
+        record_history=False):
+        """ Initialize a specie object, defaulting to a simulation time start time of 0
 
         Args:
-            species_name (:obj:`str`): the specie's name; not logically needed, but helpful for error
+            specie_name (:obj:`str`): the specie's name; not logically needed, but helpful for error
                 reporting, logging, debugging, etc.
             random_state (:obj:`numpy.random.RandomState`): a shared PRNG
-            initial_population (int): non-negative number; initial population of the specie
-            initial_flux (number, optional): initial flux for the specie; required for species whose
-                population is estimated, at least in part, by a continuous model
+            initial_population (:obj:`int`): non-negative number; initial population of the specie
+            modeled_continuously (:obj:`bool`, optional): whether a continuous submodel models this species;
+                default=`False`
+            record_history (:obj:`bool`, optional): whether to record a history of all operations; default=`False`
         """
-        assert 0 <= initial_population, '__init__(): {} population {} should be >= 0'.format(species_name, initial_population)
-        self.species_name = species_name
+        assert 0 <= initial_population, "DynamicSpecie '{}': population should be >= 0".format(specie_name)
+        # if a population is not modeled continuously then it must be a non-negative integer
+        assert modeled_continuously or float(initial_population).is_integer(), \
+            "DynamicSpecie '{}': initial discretely modeled population must be a non-negative integer, but {} isn't".format(
+                specie_name, initial_population)
+        self.specie_name = specie_name
         self.random_state = random_state
         self.last_population = initial_population
-        self.continuous_submodel = False
-        if initial_flux is not None:
-            self.continuous_submodel = True
-            self.continuous_time = 0
-            self.continuous_flux = initial_flux
+        self.modeled_continuously = modeled_continuously
+        if modeled_continuously:
+            self.population_slope = None
+            # continuous_time is None indicates that a continuous_adjustment() has not been made yet
+            self.continuous_time = None
+        self.last_adjustment_time = -float('inf')
+        self.last_read_time = -float('inf')
+        self._record_history = record_history
+        if record_history:
+            self._history = []
+            self._record_operation_in_hist(0, 'initialize', initial_population)
+
+    def _update_last_adjustment_time(self, adjustment_time):
+        """ Advance the last adjustment time to `adjustment_time`
 
-    def discrete_adjustment(self, population_change, time):
+        Args:
+            adjustment_time (:obj:`float`): the time at which the population is being adjusted
+        """
+        self.last_adjustment_time = max(self.last_adjustment_time, adjustment_time)
+
+    def _update_last_read_time(self, read_time):
+        """ Advance the last read time to `read_time`
+
+        Args:
+            read_time (:obj:`float`): the time at which the population is being read
+        """
+        self.last_read_time = max(self.last_read_time, read_time)
+
+    def _validate_adjustment_time(self, adjustment_time, method):
+        """ Raise an exception if `adjustment_time` is too early
+
+        Args:
+            adjustment_time (:obj:`float`): the time at which the population is being adjusted
+            method (:obj:`str`): name of the method making the adjustment
+
+        Raises:
+            :obj:`SpeciesPopulationError`: if `adjustment_time` is earlier than latest prior adjustment,
+                 or if adjustment_time is earlier than latest prior read
+        """
+        if adjustment_time < self.last_adjustment_time:
+            raise SpeciesPopulationError(
+                "{}(): adjustment_time is earlier than latest prior adjustment: "
+                "{:.2f} < {:.2f}".format(method, adjustment_time, self.last_adjustment_time))
+        if adjustment_time < self.last_read_time:
+            raise SpeciesPopulationError(
+                "{}(): adjustment_time is earlier than latest prior read: "
+                "{:.2f} < {:.2f}".format(method, adjustment_time, self.last_read_time))
+
+    def _validate_read_time(self, read_time, method):
+        """ Raise an exception if `read_time` is too early
+
+        Args:
+            read_time (:obj:`float`): the time at which the population is being read
+            method (:obj:`str`): name of the method making the read
+
+        Raises:
+            :obj:`SpeciesPopulationError`: if `read_time` is earlier than latest prior adjustment
+        """
+        if read_time < self.last_read_time:
+            raise SpeciesPopulationError(
+                "{}(): read_time is earlier than latest prior adjustment: "
+                "{:.2f} < {:.2f}".format(method, read_time, self.last_adjustment_time))
+
+    class Operation(Enum):
+        """ Types of operations on DynamicSpecies """
+        initialize = 1
+        discrete_adjustment = 2
+        continuous_adjustment = 3
+
+    HistoryRecord = namedtuple('HistoryRecord', 'time, operation, argument')
+    HistoryRecord.__doc__ += ': entry in a DynamicSpecie history'
+    HistoryRecord.time.__doc__ = 'simulation time of the operation'
+    HistoryRecord.operation.__doc__ = 'type of the operation'
+    HistoryRecord.argument.__doc__ = "operation's argument: initialize: population; discrete_adjustment: "\
+        "population_change; continuous_adjustment: population_slope"
+
+    def _record_operation_in_hist(self, time, method, argument):
+        """ Record a history entry
+
+        Args:
+            time (:obj:`float`): simulation time of the operation
+            method (:obj:`str`): the operation type
+            argument (:obj:`float`): the operation's argument
+        """
+        if self._record_history:
+            operation = self.Operation[method]
+            self._history.append(self.HistoryRecord(time, operation, argument))
+
+    def discrete_adjustment(self, time, population_change):
         """ Make a discrete adjustment of the specie's population
 
         A discrete-time submodel, such as the stochastic simulation algorithm (SSA), must use this
         method to adjust the specie's population.
 
         Args:
-            population_change (number): the modeled increase or decrease in the specie's population
-            time (number): the simulation time at which the predicted change occurs; this time is
+            time (:obj:`float`): the simulation time at which the predicted change occurs; this time is
                 used by `get_population` to interpolate continuous-time predictions between integrations.
+            population_change (:obj:`float`): the modeled increase or decrease in the specie's population
 
         Returns:
-            int: an integer approximation of the specie's adjusted population
+            :obj:`int`: an integer approximation of the specie's adjusted population
 
         Raises:
-            NegativePopulationError: if the predicted population at `time` is negative or
-            if decreasing the population by `population_change` would make the population negative
+            :obj:`NegativePopulationError`: if the predicted population at `time` is negative or
+                if decreasing the population by `population_change` would make the population negative
         """
+        assert float(population_change).is_integer(), \
+            "DynamicSpecie '{}': population_change must be an integer, but {} isn't".format(
+                self.specie_name, population_change)
+        self._validate_adjustment_time(time, 'discrete_adjustment')
         current_population = self.get_population(time)
         if current_population + population_change < 0:
-            raise NegativePopulationError('discrete_adjustment', self.species_name,
-                                          self.last_population, population_change)
+            raise NegativePopulationError('discrete_adjustment', self.specie_name,
+                self.last_population, population_change)
         self.last_population += population_change
+        self._update_last_adjustment_time(time)
+        self._record_operation_in_hist(time, 'discrete_adjustment', population_change)
         return self.get_population(time)
 
-    '''
-    TODO: change `continuous_adjustment` to automatically incorporate
-    population changes predicted by the flux provided by the previous `continuous_adjustment`
-    that is, use:
-        population_change = (time - self.continuous_time) * self.continuous_flux
-    '''
-    def continuous_adjustment(self, population_change, time, flux):
+    def continuous_adjustment(self, time, population_slope):
         """ A continuous-time submodel adjusts the specie's state
 
         A continuous-time submodel, such as an ordinary differential equation (ODE) or a dynamic flux
         balance analysis (FBA) model, uses this method to adjust the specie's state. Each
         integration of a continuous-time model must predict a specie's population change and the
-        population's short-term future rate of change, i.e., its `flux`. Further, since an
-        integration of a continuous-time model at the current time must depend on this specie's
-        population just before the integration, we assume that `population_change` incorporates
-        population changes predicted by the flux provided by the previous `continuous_adjustment`
-        call.
+        population's short-term future rate of change.
 
         Args:
-            population_change (number): modeled increase or decrease in the specie's population
-            time (number): the simulation time at which the predicted change occurs; this time is
+            time (:obj:`float`): the simulation time at which the predicted change occurs; this time is
                 used by `get_population` to interpolate continuous-time predictions between
                 integrations.
-            flux (number): the predicted flux of the specie at the provided time
+            population_slope (:obj:`float`): the predicted rate of change of the specie at the provided time
 
         Returns:
-            int: the specie's adjusted population, rounded to an integer
+            :obj:`int`: the specie's adjusted population, rounded to an integer
 
         Raises:
-            :obj:`SpeciesPopulationError`: if an initial flux was not provided
-            :obj:`SpeciesPopulationError`: if `time` is not greater than the time of the most recent
-                `continuous_adjustment` call on this `specie`
-            :obj:`NegativePopulationError`: if applying `population_change` makes the population go negative
+            :obj:`SpeciesPopulationError`: if an initial population slope was not provided, or
+                if `time` is not greater than the time of the most recent `continuous_adjustment` call
+            :obj:`NegativePopulationError`: if updating the population based on the previous `population_slope`
+                makes the population go negative
         """
-        if not self.continuous_submodel:
-            raise SpeciesPopulationError("continuous_adjustment(): initial flux was not provided")
-        # the simulation time must advance between adjacent continuous adjustments
-        if time <= self.continuous_time:
-            raise SpeciesPopulationError("continuous_adjustment(): time <= self.continuous_time: "
-                                         "{:.2f} < {:.2f}".format(time, self.continuous_time))
-        if self.last_population + population_change < 0:
-            raise NegativePopulationError('continuous_adjustment', self.species_name,
-                                          self.last_population, population_change, time-self.continuous_time)
+        if not self.modeled_continuously:
+            raise SpeciesPopulationError(
+                "continuous_adjustment(): DynamicSpecie not modeled by a continuous submodel; "
+                "set modeled_continuously True")
+        self._validate_adjustment_time(time, 'continuous_adjustment')
+        # self.continuous_time is None until the first continuous_adjustment()
+        if self.continuous_time is not None:
+            if self.last_population + self.population_slope * (time - self.continuous_time) < 0:
+                raise NegativePopulationError("continuous_adjustment: "
+                    "population_slope: {}, continuous_time: {}".format(
+                    self.population_slope, self.continuous_time),
+                    self.specie_name, self.last_population, self.population_slope * (time - self.continuous_time))
+            # add the population change since the last continuous_adjustment
+            self.last_population += self.population_slope * (time - self.continuous_time)
         self.continuous_time = time
-        self.continuous_flux = flux
-        self.last_population += population_change
+        self.population_slope = population_slope
+        self._update_last_adjustment_time(time)
+        self._record_operation_in_hist(time, 'continuous_adjustment', population_slope)
         return self.get_population(time)
 
-    def get_population(self, time=None):
+    def get_population(self, time, interpolate=None, round=True):
         """ Provide the specie's current population
 
         If one of the submodel(s) predicting the specie's population is a continuous-time model,
-        then use the specie's last flux to interpolate the current population, as described in the
-        class documentation.
+        then use the specie's last `population_slope` to interpolate the current population, as
+        described in the class documentation.
 
         Clearly, species populations in biological systems are non-negative integers. However,
         continuous-time models approximate populations with continuous representations, and
         therefore predict real, non-integral, populations. But discrete-time models like SSA
         do not naturally handle non-integral copy numbers.
 
         We resolve this conflict by storing real valued populations within a specie, but
         providing only integral population predictions. To aovid the bias that would arise by
         always using floor() or ceiling() to convert a float to an integer, population predictions
         are stochastically rounded before being returned by `get_population`. *This means
         that a sequence of calls to `get_population` which do not have any interveening
         adjustment operations may **NOT** return a sequence of equal population values.*
 
         Args:
-            time (:obj:`Rational`, optional): the current simulation time; `time` is required if one of the
-                submodels modeling the specie is a continuous-time submodel.
+            time (:obj:`float`): the current simulation time
+            round (:obj:`bool`, optional): if `round` then round the populations to integers
+            interpolate (:obj:`bool`, optional): if not `None` then control interpolation;
+                otherwise it's controlled by the 'interpolate' config variable
 
         Returns:
-            int: an integer approximation of the specie's adjusted population
+            :obj:`int`: if `round`, an integer approximation of the specie's population, otherwise
+                the floating population
 
         Raises:
-            :obj:`SpeciesPopulationError`: if `time` is required but not provided
-            :obj:`SpeciesPopulationError`: if `time` is earlier than the time of a previous continuous adjustment
-            NegativePopulationError: if interpolation predicts a negative population
+            :obj:`SpeciesPopulationError`: if `time` is earlier than the time of a previous continuous
+                adjustment or discrete adjustment
+            :obj:`NegativePopulationError`: if interpolation predicts a negative population
         """
-        if not self.continuous_submodel:
-            return self.random_state.round(self.last_population)
+        self._validate_read_time(time, 'get_population')
+        if not self.modeled_continuously:
+            self._update_last_read_time(time)
+            # self.last_population does not need to be rounded as discrete submodels only change
+            # populations by integral amounts
+            return self.last_population
         else:
-            if time is None:
-                raise SpeciesPopulationError("get_population(): time needed because "
-                                             "continuous adjustment received at time {:.2f}".format(self.continuous_time))
-            if time < self.continuous_time:
-                raise SpeciesPopulationError("get_population(): time < self.continuous_time: {:.2f} < {:.2f}\n".format(
-                    time, self.continuous_time))
             interpolation = 0
-            if config_multialgorithm['interpolate']:
-                interpolation = (time - self.continuous_time) * self.continuous_flux
-            if self.last_population + interpolation < 0:
-                raise NegativePopulationError('get_population', self.species_name,
-                                              self.last_population, interpolation, time - self.continuous_time)
+            if self.continuous_time is not None:
+                if interpolate is None:
+                    interpolate = config_multialgorithm['interpolate']
+                if interpolate:
+                    interpolation = (time - self.continuous_time) * self.population_slope
+                    if self.last_population + interpolation < 0:
+                        raise NegativePopulationError('get_population', self.specie_name,
+                            self.last_population, interpolation, time - self.continuous_time)
             float_copy_number = self.last_population + interpolation
-            return self.random_state.round(float_copy_number)
+            self._update_last_read_time(time)
+            # if round then round the return value to an integer, otherwise don't
+            if round:
+                # this cannot return a negative number
+                return self.random_state.round(float_copy_number)
+            return float_copy_number
+
+    def get_history(self):
+        """ Obtain this `DynamicSpecie`'s history
+
+        Returns:
+            :obj:`list`: a list of `HistoryRecord`s, ordered by time
+
+        Raises:
+            :obj:`SpeciesPopulationError`: if the history wasn't recorded
+        """
+        if self._record_history:
+            return self._history
+        else:
+            raise SpeciesPopulationError('history not recorded')
 
     def __str__(self):
-        if self.continuous_submodel:
-            return "species_name: {}; last_population: {}; continuous_time: {}; continuous_flux: {}".format(
-                self.species_name, self.last_population, self.continuous_time, self.continuous_flux)
+        if self.modeled_continuously:
+            return "specie_name: {}; last_population: {}; continuous_time: {}; population_slope: {}".format(
+                self.specie_name, self.last_population, self.continuous_time, self.population_slope)
         else:
-            return "species_name: {}; last_population: {}".format(
-                self.species_name, self.last_population)
+            return "specie_name: {}; last_population: {}".format(self.specie_name, self.last_population)
 
     @staticmethod
     def heading():
         """ Return a heading for a tab-separated table of species data """
-        return '\t'.join('species_name last_population continuous_time continuous_flux'.split())
+        return '\t'.join('specie_name last_population continuous_time population_slope'.split())
 
     def row(self):
         """ Return a row for a tab-separated table of species data """
-        if self.continuous_submodel:
-            return "{}\t{:.2f}\t{:.2f}\t{:.2f}".format(self.species_name, self.last_population,
-                                                       self.continuous_time, self.continuous_flux)
+        if self.modeled_continuously:
+            if self.continuous_time is None:
+                return "{}\t{:.2f}".format(self.specie_name, self.last_population)
+            else:
+                return "{}\t{:.2f}\t{:.2f}\t{:.2f}".format(self.specie_name, self.last_population,
+                self.continuous_time, self.population_slope)
         else:
-            return "{}\t{:.2f}".format(self.species_name, self.last_population)
+            return "{}\t{:.2f}".format(self.specie_name, self.last_population)
