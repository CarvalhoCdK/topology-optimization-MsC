from openmdao.api import ExplicitComponent
import numpy as np


class BucklingConstraint(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('eigenvalue_gap', types=float)
        self.options.declare('max_buckling_load', types=float)
        self.options.declare('number_of_eigenvalues', types=int)
        self.options.declare('data_recorder', default=False)


    def setup(self):
        neig = self.options['number_of_eigenvalues']

        self.add_input('eigenvalues', shape=neig)

        self.add_output('residuals', shape=neig)

        ar = np.arange(neig)
        self.declare_partials('residuals', 'eigenvalues', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        neig = self.options['number_of_eigenvalues']
        gap = self.options['eigenvalue_gap']
        p_max = self.options['max_buckling_load']

        residuals = np.zeros(neig)
        for n in range(neig):
            mu = inputs['eigenvalues'][n]

            residuals[n] = mu * p_max * gap**(-n) + 1
                   
        
        outputs['residuals'] = residuals


    def compute_partials(self, inputs, partials):
        neig = self.options['number_of_eigenvalues']
        gap = self.options['eigenvalue_gap']
        p_max = self.options['max_buckling_load']

        ## Recording information
        data_recorder = self.options['data_recorder']
        if data_recorder != False:
            data_recorder.get_buckling_constraints(inputs['eigenvalues'])
        print('Eigenvalues')
        print(inputs['eigenvalues'])

        derivs = np.zeros(neig)
        for n in range(neig):
            derivs[n] = p_max * gap**(-n)

        partials['residuals', 'eigenvalues'] = derivs

