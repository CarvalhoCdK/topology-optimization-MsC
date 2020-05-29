from openmdao.api import ExplicitComponent

import numpy as np


class PenalizationComp(ExplicitComponent):
    """
    """
    def initialize(self):
        self.options.declare('penal', types=(int, float))
        self.options.declare('n_elements', types=int)


    def setup(self):
        nelements = self.options['n_elements']
        self.it = -1
        self.cont_it = 20
        self.p = 0.8


        self.add_input('densities', shape=nelements)
        self.add_output('multipliers', shape=nelements)

        ar = np.arange(nelements)

        self.declare_partials('multipliers', 'densities', rows=ar, cols=ar)
          

    def compute(self, inputs, outputs):
        penal = self.options['penal']
        
        # if self.it < self.cont_it:
        #     p = self.p
        #     self.p += (penal-1)/self.cont_it
        # else:
        #     p = penal
        
        # self.it += 1
        # #print('penal: ' + str(p))
        # print('it:' + str(self.it))

        outputs['multipliers'] = inputs['densities'] ** penal
        


    def compute_partials(self, inputs, partials):
        p = self.options['penal']
       # print('penal derivs' + str(p))

        partials['multipliers', 'densities'] = p * inputs['densities'] ** (p-1)
