from openmdao.api import ExplicitComponent

import numpy as np


class VolumeComp(ExplicitComponent):
    """
    """
    def initialize(self):
        self.options.declare('nelements', types=int)
        self.options.declare('data_recorder', default=False)


    def setup(self):
        nelements = self.options['nelements']

        self.add_input('densities', shape=nelements)
        self.add_output('volume')
        
        self.declare_partials('volume', 'densities')
        # Finite difference all partials.
        #self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        nelements = self.options['nelements']
        

        volume = sum(inputs['densities']) / nelements
        self.volume = volume

        #
        #

        outputs['volume'] = volume


    def compute_partials(self, inputs, partials):
        nelements = self.options['nelements']

        ## Recording information
        data_recorder = self.options['data_recorder']
        if data_recorder != False:
            data_recorder.get_volume(self.volume)

        partials['volume', 'densities'] = 1 / nelements