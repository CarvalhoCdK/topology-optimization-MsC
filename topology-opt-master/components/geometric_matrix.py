from openmdao.api import ExplicitComponent

from fem.fe_model import FEA_model
from scipy.sparse import coo_matrix, csc_matrix
import numpy as np


class GeometricMatrixComp(ExplicitComponent):
    """
    """

    def initialize(self):
        self.options.declare('fe_model')
        self.options.declare('n_dof', types=int)
        self.options.declare('unknown_dof', types=int)

    def setup(self):
        unknown_dof = self.options['unknown_dof']
        ndof = self.options['n_dof']

        self.add_input('displacements', shape=(unknown_dof))
        self.add_output('geometric_matrix', shape=(ndof, ndof))

        self.declare_partials('geometric_matrix', 'displacements', method='fd')


    def compute(self, inputs, outputs):
        fe = self.options['fe_model']
        bu = fe.bu
        ndof = self.options['n_dof']

        u = np.zeros(ndof)
        u[bu] = inputs['displacements']

        kg = fe.assemble_kg(u).toarray()

        outputs['geometric_matrix'] = kg

