'''Define multi-algoritmic simulation errors.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-12
:Copyright: 2016, Karr Lab
:License: MIT
'''

class Error(Exception):
    '''Base class for exceptions involving multi-algoritmic simulation.'''
    pass

class SpeciesPopulationError(Error):
    '''Exception raised when species population management encounters a problem.'''

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        '''Provide the Exception's msg; needed for Python 2.7, although not documented.'''
        return self.msg

class NegativePopulationError(Error):
    '''Exception raised when a negative specie population is predicted.

    The sum of `last_population` and `population_decrease` equals the predicted negative population.

    Attributes:
        method (:obj:`str`): name of the method in which the exception occured
        specie (:obj:`str`): name of the specie whose population is predicted to be negative
        last_population (:obj:`float`): previous recorded population for the specie
        population_decrease (:obj:`float`): change to the population which would make it negative
        delta_time (:obj:`float`, optional): if the specie has been updated by a continuous submodel,
            time since the last continuous update
    '''
    def __init__(self, method, specie, last_population, population_decrease, delta_time=None):
        self.method=method
        self.specie=specie
        self.last_population=last_population
        self.population_decrease=population_decrease
        self.delta_time=delta_time

    def __eq__(self, other):
        '''Determine whether two instances have the same content'''
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.method, self.specie, self.last_population, self.population_decrease,
            self.delta_time))

    def __str__(self):
        rv = "{}(): negative population predicted for '{}', with decline from {:g} to {:g}".format(
            self.method, self.specie, self.last_population,
            self.last_population+self.population_decrease)
        if self.delta_time is None:
            return rv
        else:
            if self.delta_time == 1:
                return rv + " over {:g} time unit".format(self.delta_time)
            else:
                return rv + " over {:g} time units".format(self.delta_time)
            