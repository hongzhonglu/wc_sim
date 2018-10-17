""" A sub-model that employs a system of ordinary differential equations (ODEs) to model a set of reactions.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-10-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

import numpy as np
from scipy.constants import Avogadro
from scikits.odes import ode

from wc_sim.core.config import core as config_core_core
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.config import core as config_core_multialgorithm
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError

config_core = config_core_core.get_config()['wc_sim']['core']
config_multialgorithm = \
    config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


'''
ODE concepts
    convert pops to concentrations
    pick time step
    solve with cvode
        check results
    update continuous pops
ODE practical
    install and test cvode
    cvode installed in docker
    reuse existing use of cvode?
    ODE in intro to wc modeling
'''

'''
todos:
    setup prerequisites for Odes:
        instructions:
            prerequisites (dependencies) and my notes
                numpy (automatically dealt with if using pip >=10)
                    [already installed]
                Python header files (python-dev/python3-dev on Debian/Ubuntu-based distributions, python-devel on Fedora)
                C compiler
                    [already installed, assuming gcc version 7.3.0 (Ubuntu 7.3.0-16ubuntu3) is OK]
                Fortran compiler (e.g. gfortran)
                Sundials 2.7.0
                    [SUNDIALS 2.7.0 was released 2016-09-26; 3.2.0 is the current (2018-09-28) production release; can it be used?;
                        current (2018-10-12) GitHub version of Odes uses SUNDIALS 3.1.1]
            install OpenBLAS
                $ sudo apt update
                $ apt search openblas
                $ sudo apt install libopenblas-dev
                $ sudo update-alternatives --config libblas.so.3
            install odes
    setup Docker for Odes:
        commands (skip sudo, as container user is root):
            apt install python3-dev --yes

            # install gfortran
            apt install gfortran --yes

            # install OpenBLAS
            apt update
            # apt search openblas
            apt install libopenblas-dev --yes
            # skip, doesn't work: update-alternatives --config libblas.so.3

            # SUNDIALS 2.7.0
            mkdir /tmp/sundials
            pushd /tmp/sundials > /dev/null
            wget https://computation.llnl.gov/projects/sundials/download/sundials-2.7.0.tar.gz
            tar xzf sundials-2.7.0.tar.gz
            mkdir instdir
            mkdir builddir
            cd builddir
            cmake \
            -DEXAMPLES_ENABLE=OFF \
            -DCMAKE_C_COMPILER=/usr/bin/gcc \
            -DLAPACK_ENABLE=ON \
            ../sundials-2.7.0
            make install
            popd > /dev/null
            # todo: test SUNDIALS

            # install odes
            pip install scikits.odes
        update wc-env (name?) Docker image for Odes:
            add Odes prerequisites commands to wc-env Dockerfile
            test it

    test Odes using its jupyter notebook
    finish writing this module
    write test_odes.py
    use some SMBL 'semantic' tests:
        review some
        choose some to use
        expand validate.py to use them
        use them
'''

class OdeSubmodel(DynamicSubmodel):
    """

    Attributes:
        rate_of_change_expressions (:obj:`list`): a list of coefficient, rate law tuples for each species
        solver (:obj:`scikits.odes.ode.ode`): the Odes ode solver
        # todo: add attributes made by set_up_optimizations() and elsewhere
    """

    # register the message types sent by OdeSubmodel
    messages_sent = [message_types.RunOde]

    # register 'handle_RunOde_msg' to handle RunOde events
    event_handlers = [(message_types.RunOde, 'handle_RunOde_msg')]

    # prevent simultaneous use of multiple solver instances because of the 'OdeSubmodel.instance = self'
    # also, it's unclear whether that works; see: https://stackoverflow.com/q/34291639
    # todo: enable simultaneous use of multiple OdeSubmodel instances
    using_solver = False

    def __init__(self, id, dynamic_model, reactions, species, parameters, dynamic_compartments,
        local_species_population, time_step):
        """ Initialize an ODE submodel instance.

        Args:
            id (:obj:`str`): unique id of this dynamic ODE submodel
            dynamic_model (:obj: `DynamicModel`): the aggregate state of a simulation
            reactions (:obj:`list` of `wc_lang.Reaction`): the reactions modeled by this ODE submodel
            species (:obj:`list` of `wc_lang.Species`): the species that participate in the reactions modeled
                by this ODE submodel, with their initial concentrations
            parameters (:obj:`list` of `wc_lang.Parameter`): the model's parameters
            dynamic_compartments (:obj: `dict`): `DynamicCompartment`s, keyed by id, that contain
                species which participate in reactions that this ODE submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                ODE submodel's species population
            time_step (:obj:`float`): time interval between ODE analyses
        """
        super().__init__(id, dynamic_model, reactions, species, parameters, dynamic_compartments,
            local_species_population)
        if time_step <= 0:
            raise MultialgorithmError("OdeSubmodel {}: time_step must be positive, but is {}".format(
                self.id, time_step))
        if 1 < len(self.dynamic_compartments):
            raise MultialgorithmError("OdeSubmodel {}: multiple compartments not supported".format(self.id))
        self.time_step = time_step
        self.set_up_optimizations()
        self.set_up_ode_submodel()

    def set_up_optimizations(self):
        """For optimization, pre-compute and pre-allocate data structures"""
        self.species_ids = [specie.get_id() for specie in self.species]
        # make fixed set of species ids used by this OdeSubmodel
        self.species_ids_set = set(self.species_ids)
        # pre-allocate dict of adjustments for LocalSpeciesPopulation
        self.adjustments = {species_id:None for species_id in self.species_ids}
        # pre-allocate np arrays for concentrations and populations
        self.concentrations = np.zeros(len(self.species_ids))
        self.populations = np.zeros(len(self.species_ids))

    def get_compartment_id(self):
        # todo: discard when multiple compartments supported
        return list(self.dynamic_compartments.keys())[0]

    def get_compartment_volume(self):
        # todo: discard when multiple compartments supported
        return self.compute_volumes()[self.get_compartment_id()]

    def get_concentrations(self):
        """Get current shared concentrations in numpy array"""
        specie_concentrations_dict = self.get_specie_concentrations()
        np.copyto(self.concentrations,
            [specie_concentrations_dict[id] for id in self.species_ids])
        return self.concentrations

    def concentrations_to_populations(self, concentrations):
        """Convert numpy array of concentrations to array of populations"""
        # todo: move this to a utility
        # optimization: copy concentrations to existing self.populations &
        # modify self.populations in place with *= and out=
        np.copyto(self.populations, concentrations)
        vol_avo = self.get_compartment_volume() * Avogadro
        self.populations *= vol_avo
        return np.rint(self.populations, out=self.populations)

    def set_up_ode_submodel(self):
        """Set up an ODE submodel, including its ODE solver"""

        # HACK!: store this instance in OdeSubmodel class variable, so that right_hand_side() can use it
        OdeSubmodel.instance = self
        # disable locking temporarily
        # self.get_solver_lock()

        # this is optimal, but costs O(|self.reactions| * |rxn.participants|)
        tmp_rate_of_change_expressions = {species_id:[] for species_id in self.species_ids}
        for idx, rxn in enumerate(self.reactions):
            for species_coefficient in rxn.participants:
                dyn_reaction = self.dynamic_reactions[idx]
                if not hasattr(dyn_reaction, 'dynamic_rate_law'):
                    raise MultialgorithmError("OdeSubmodel {}: dynamic reaction '{}' needs dynamic rate law".format(
                        self.id, dyn_reaction.id))
                species_id = species_coefficient.species.get_id()
                tmp_rate_of_change_expressions[species_id].append((species_coefficient.coefficient,
                    dyn_reaction.dynamic_rate_law))

        self.rate_of_change_expressions = []
        for species_id in self.species_ids:
            self.rate_of_change_expressions.append(tmp_rate_of_change_expressions[species_id])

    def get_solver_lock(self):
        cls = self.__class__
        if not cls.using_solver:
            cls.using_solver = True
            return True
        else:
            raise MultialgorithmError("OdeSubmodel {}: cannot get_solver_lock".format(self.id))

    def release_solver_lock(self):
        cls = self.__class__
        # todo: need a mechanism for scheduling an event that calls this
        cls.using_solver = False
        return True

    def set_up_ode_solver(self):
        """Set up the `scikits.odes` ODE solver"""
        # todo: methods in DynamicSubmodel and LocalSpeciesPopulation to put concentrations directly
        # into existing np arrays
        specie_concentrations_dict = self.get_specie_concentrations()
        self.concentrations = np.asarray([specie_concentrations_dict[id] for id in self.species_ids])
        # use CVODE from LLNL's SUNDIALS (https://computation.llnl.gov/projects/sundials)
        self.solver = ode('cvode', self.right_hand_side, old_api=False)
        solver_return = self.solver.init_step(self.time, self.concentrations)
        if not solver_return.flag:
            raise MultialgorithmError("OdeSubmodel {}: solver.init_step() failed: '{}'".format(self.id,
                solver_return.message)) # pragma: no cover
        return solver_return

    @staticmethod
    def right_hand_side(time, concentrations, concentration_change_rates, testing=False):
        """Evaluate concentration change rates for all species; called by ODE solver

        Args:
            time (:obj:`float`): simulation time
            concentrations (:obj:`numpy.ndarray`): concentrations of species at time `time`, in the
                same order as `self.species`
            concentration_change_rates (:obj:`numpy.ndarray`): the rate of change of concentrations at
                time `time`; written by this method
            testing (:obj:`bool`, optional): if set, raise exception to help testing

        Returns:
            :obj:`int`: return 0 to indicate success, 1 to indicate failure;
                see http://bmcage.github.io/odes/version-v2.3.2/ode.html#scikits.odes.ode.ode;
                but the important side effects are the values in `concentration_change_rates`
        """
        # this is called by a c code wrapper that's used by the CVODE ODE solver
        try:
            # obtain hacked instance reference
            self = OdeSubmodel.instance

            # for each specie in `concentrations` sum evaluations of rate laws in self.rate_of_change_expressions
            parameter_values = self.get_parameter_values()
            species_concentrations = self.get_specie_concentrations()
            # todo: change when multiple compartments supported
            volume = self.get_compartment_volume()

            for idx, conc in enumerate(np.nditer(concentrations)):
                specie_rxn_rates = []
                for coeff, dyn_rate_law in self.rate_of_change_expressions[idx]:
                    specie_rxn_rates.append(coeff * dyn_rate_law.eval(
                        self.time,
                        parameter_values=parameter_values,
                        species_concentrations=species_concentrations,
                        compartment_volume=volume))
                concentration_change_rates[idx] = sum(specie_rxn_rates)
            return 0
        except Exception as e:
            if testing:
                raise MultialgorithmError("OdeSubmodel {}: solver.right_hand_side() failed: '{}'".format(
                    self.id, e))
            return 1

    def run_ode_solver(self):
        """Run the ODE solver for one time step and save its results"""
        ### run the ODE solver
        # re-initialize the solver to include changes in concentrations by other submodels
        self.set_up_ode_solver()
        # todo: minimize round-off error for time by counting steps and using multiplication
        end_time = self.time + self.time_step
        solution = self.solver.step(end_time)
        if solution.errors.t:
            raise MultialgorithmError("OdeSubmodel {}: odes solver error: '{}' at time {}".format(
                self.id, solution.message, solution.errors.t))
        
        print('solution.values.y.shape', solution.values.y.shape)
        ### store results in local_species_population
        '''
        approach
            pre-compute mean rate of change for the next time step
            init_pops = initial population at start of this ODE analysis
                solution.values.y is an np array w shape 1xnum(species)
            curr_pops = solution.values.y converted to pops
            pops_change = curr_pops - init_pops

            rate = pops_change/self.time_step
            map all to dict (pre-allocated)
        '''
        # todo: optimization: optimize LocalSpeciesPopulation to provide an array
        init_pops_dict = self.local_species_population.read(self.time, self.species_ids_set)
        # todo: optimization: after optimizing LocalSpeciesPopulation, optimize these
        init_pops_list = [init_pops_dict[species_id] for species_id in self.species_ids]
        init_pops_array = np.array(init_pops_list)

        # convert concentrations to populations
        curr_pops = self.concentrations_to_populations(solution.values.y)
        print('curr_pops, init_pops_array', curr_pops, init_pops_array)
        pops_change = curr_pops - init_pops_array

        rate = pops_change / self.time_step
        print('rate', rate)
        for idx, species_id in enumerate(self.species_ids):
            self.adjustments[species_id] = (0, rate[idx])
        # todo: optimization: optimize LocalSpeciesPopulation to accept arrays
        self.local_species_population.adjust_continuously(self.time, self.adjustments)

        '''
        first draft:
        if self.adjust_continuously:
            rate = pops_change/self.time_step
            for idx, species_id in enumerate(self.species_ids):
                self.adjustments[species_id][0] = pops_change[idx]
                self.adjustments[species_id][1] = rate[idx]
            self.local_species_population.adjust_continuously(self.time, self.adjustments)

        else:
            # adjust discretely
            for idx, species_id in enumerate(self.species_ids):
                self.adjustments[species_id] = pops_change[idx]
            self.local_species_population.adjust_discretely(self.time, self.adjustments)
        '''

    # schedule and handle events
    def send_initial_events(self):
        """Send this ODE submodel's initial event"""
        self.schedule_next_ode_analysis()

    def schedule_next_ode_analysis(self):
        """Schedule the next analysis by this ODE submodel"""
        # todo: count events to avoid round off error; I think Checkpointing does
        self.send_event(self.time_step, self, message_types.RunOde())

    def handle_RunOde_msg(self):
        """Handle a RunOde message"""
        self.run_ode_solver()
        self.schedule_next_ode_analysis()
