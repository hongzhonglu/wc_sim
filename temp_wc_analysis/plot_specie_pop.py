"""Plot a species population's dynamics. Inputs logging log.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-07-12
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import sys
import math
import argparse

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

from past import autotranslate
autotranslate(['recordtype'])
from recordtype import recordtype

'''
Steps:
read and parse log
plot parameters:
    line weight
    color, shape, size of symbols
    location legend
x is time, and y is copy number
'''

# A population adjustment event
event_fields = 'time adjustment_type population flux'
Adjustment_event = recordtype( 'Adjustment_event', event_fields )

def convert_floats( l ):
    """Convert all string which represent floats into floats. Changes list l in place."""

    for index in range(len(l)):
        try:
            l[index] = float(l[index])
        except:
            pass

def print_event_list(el):
    print( '\t'.join( event_fields.split() ) )
    for e in el:
        print( '\t'.join( [ str(e._asdict()[f]) for f in event_fields.split() ] ))

class PlotPopulationDynamics(object):

    @staticmethod
    def parse_cli_args():
        parser = argparse.ArgumentParser( description="Plot a species population's dynamics. "
            "Inputs logging log." )
        parser.add_argument( 'log_file', type=str, help="A log of species population adjustments")
        parser.add_argument( '--pdf_file', '-p', type=str, help='The output pdf file.' )
        args = parser.parse_args()
        return args

    @staticmethod
    def parse_log( file ):
        """Parse a species logging log.

        The log's format is determined by the plot handlers in .cfg files.

        Arguments:
            file: string; the log file

        Returns:
            list of logged events
        """
        events = []
        try:
            fh = open( file, 'r' )
            fh.readline()   # discard header line
            for line in fh.readlines():
                line = line.strip()
                (junk, data) = line.split('#')
                data = data.split('\t')
                convert_floats( data )
                events.append( Adjustment_event( *data ) )
            fh.close()
            return events
        except IOError:
            sys.exit( "Error: cannot read {} \n".format( file ) )


    @staticmethod
    def get_line_segment_endpoints( start_event, end_event ):
        """Determine the plot line segment endpoints between a pair of events.

            Given:
                flux = last flux value
                pop = population following current adjustment
                time_cur = time of current adjustment
                time_next = time of next adjustment

                line segment: ( time_cur, pop ) - ( time_next, pop + (time_next - time_cur )*flux )

            Returns:
                Return ((x_start, x_end), (y_start, y_end) ), as used by matplotlib
        """
        flux = start_event.flux
        if flux == None or flux == 'None':
            flux = 0
        return  ( ( start_event.time, end_event.time),
            ( start_event.population, start_event.population + ( end_event.time-start_event.time)*flux) )

    @staticmethod
    def plot_species_population_dynamics( args, events ):
        """Plot a species population's dynamics.
        """

        # plot markers
        continuous_events = filter( lambda x: x.adjustment_type == 'continuous_adjustment', events )
        discrete_events = filter( lambda x: x.adjustment_type == 'discrete_adjustment', events )
        continuous_params = { 'marker' : 'o',
            'color' : 'red',
            'label' : 'Continuous adjustments' }
        discrete_params = { 'marker' : '^',
            'color' : 'green',
            'label' : 'Discrete adjustments' }

        pyplot.scatter(
            map( lambda e: e.time, continuous_events ),
            map( lambda e: e.population, continuous_events ),
            **continuous_params )
        pyplot.scatter(
            map( lambda e: e.time, discrete_events ),
            map( lambda e: e.population, discrete_events ),
            **discrete_params )
        pyplot.legend()

        # plot piece-wise linear line segments
        data = []
        for index in range( len( events ) - 1 ):
            (x_vals, y_vals) = PlotPopulationDynamics.get_line_segment_endpoints( events[index], events[index+1] )
            # print(x_vals, y_vals)
            data.append(x_vals)
            data.append(y_vals)
            data.append('blue')
        pyplot.plot(*data)

        # expand axes by 1 in each direction, so all data shows
        ymin, ymax = pyplot.ylim()
        pyplot.ylim( ymin-1, ymax+1)
        xmin, xmax = pyplot.xlim()
        pyplot.xlim( xmin-1, xmax+1)
        pyplot.xlabel('Simulated time (units here)')
        pyplot.ylabel('Predicted copy number')

    @staticmethod
    def output_plot( args ):
        if args.pdf_file:
            print( "Writing '{}'.".format( args.pdf_file ) )
            fig = pyplot.gcf()
            fig.savefig(args.pdf_file)
            pyplot.close(fig)
        else:
            pyplot.show()

    @staticmethod
    def main():
        args = PlotPopulationDynamics.parse_cli_args()
        event_list = PlotPopulationDynamics.parse_log( args.log_file )
        PlotPopulationDynamics.plot_species_population_dynamics( args, event_list )
        PlotPopulationDynamics.output_plot( args )

if __name__ == '__main__':
    try:
        # TODO(Arthur): important: plot multiple species simultaneously, either on one plot and/or in a grid
        # TODO(Arthur): time units
        # TODO(Arthur): important: label with species name
        # TODO(Arthur): command line control of markers, colors, labels, etc.

        PlotPopulationDynamics.main()
    except KeyboardInterrupt:
        pass

