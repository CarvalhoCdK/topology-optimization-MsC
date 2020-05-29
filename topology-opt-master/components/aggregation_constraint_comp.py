from openmdao.api import ExplicitComponent
import numpy as np


class AggregationConstraint(ExplicitComponent):
    """
    Buckling constraint modeled with an aggregation approach.
    Uses p-norm.
    """

    def initialize(self):
        self.options.declare('aggregation_parameter', types=float)
        self.options.declare('max_buckling_load', types=float)
        self.options.declare('number_of_eigenvalues', types=int)


    def setup(self):
        neig = self.options['number_of_eigenvalues']

        self.add_input('eigenvalues', shape=neig)
        self.add_output('residuals')

        self.declare_partials('residuals', 'eigenvalues')


    def compute(self, inputs, outputs):
        neig = self.options['number_of_eigenvalues']
        p_norm = self.options['aggregation_parameter']
        p_max = self.options['max_buckling_load']

        
        for n in range(neig):
            mp_i = (-inputs['eigenvalues'][n])**p_norm      

        print('Eigenvalues')
        print(inputs['eigenvalues'])
        outputs['residuals'] = p_max * np.sum(mp_i)**(1/p_norm)


    def compute_partials(self, inputs, partials):
        neig = self.options['number_of_eigenvalues']
        p_max = self.options['max_buckling_load']
        p_norm = self.options['aggregation_parameter']

        derivs = np.zeros(neig)
        for n in range(neig):
            mp_i = (-inputs['eigenvalues'][n])**p_norm

            mp_j = inputs['eigenvalues'][n]**(p_norm - 1)

        derivs = p_max * -np.sum(mp_i)**(1 - p_norm) * np.sum(mp_j)

        partials['residuals', 'eigenvalues'] = derivs

