'''Analysis utility functions.

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2016-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

# TODO(Arthur): IMPORTANT: refactor and replace

from matplotlib import pyplot
from matplotlib import ticker
from wc_lang import Model, Submodel
from scipy.constants import Avogadro
import numpy as np
import re

def plot(model, time = np.zeros(0),
    species_counts = None, volume = np.zeros(0), extracellular_volume = np.zeros(0),
    selected_species_compartments = [],
    yDatas = {},
    units = 'mM', title = '', fileName = ''):

    #convert time to hours
    time = time.copy() / 3600

    #create figure
    fig = pyplot.figure()

    #extract data to plot
    if not yDatas:
        yDatas = {}
        for species_compartment_id in selected_species_compartments:
            #extract data
            match = re.match('^(?P<speciesId>[a-z0-9\-_]+)\[(?P<compartmentId>[a-z0-9\-_]+)\]$',
                species_compartment_id, re.I).groupdict()
            speciesId = match['speciesId']
            compartmentId = match['compartmentId']

            if isinstance(model, Model):
                species = model.get_component_by_id(speciesId, 'species')
                compartment = model.get_component_by_id(compartmentId, 'compartments')
                yData = species_counts[species.index, compartment.index, :]
            elif isinstance(model, Submodel):
                yData = species_counts[species_compartment_id]
            else:
                raise Exception('Invalid model type %s' % model.__class__.__name__)

            #scale
            if compartmentId == 'c':
                V = volume
            else:
                V = extracellular_volume

            if units == 'pM':
                scale = 1 / Avogadro / V * 1e12
            elif units == 'nM':
                scale = 1 / Avogadro / V * 1e9
            elif units == 'uM':
                scale = 1 / Avogadro / V * 1e6
            elif units == 'mM':
                scale = 1 / Avogadro / V * 1e3
            elif units == 'M':
                scale = 1 / Avogadro / V * 1e0
            elif units == 'molecules':
                scale = 1
            else:
                raise Exception('Invalid units "%s"' % units)

            yData *= scale

            yDatas[species_compartment_id] = yData

    #plot results
    yMin = 1e12
    yMax = -1e12
    for label, yData in yDatas.items():
        #update range
        yMin = min(yMin, np.min(yData))
        yMax = max(yMax, np.max(yData))

        #add to plot
        pyplot.plot(time, yData, label=label)

    #set axis limits
    pyplot.xlim((0, time[-1]))
    pyplot.ylim((yMin, yMax))

    #add axis labels and legend
    if title:
        pyplot.title(title)

    pyplot.xlabel('Time (h)')

    if units == 'molecules':
        pyplot.ylabel('Copy number')
    else:
        pyplot.ylabel('Concentration (%s)' % units)

    y_formatter = ticker.ScalarFormatter(useOffset=False)
    pyplot.gca().get_yaxis().set_major_formatter(y_formatter)

    if len(selected_species_compartments) > 1:
        pyplot.legend()

    #save
    if fileName:
        fig.savefig(fileName)
        pyplot.close(fig)
