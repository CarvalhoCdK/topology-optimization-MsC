from openmdao.api import ExplicitComponent

from fem.fe_model import FEA_model
from scipy.sparse import coo_matrix, csc_matrix
import numpy as np


class StiffnessMatrixComp(ExplicitComponent):
    """
    Computing list containing all individual values form the stiffness matrix in 
    a 2D array.
    """

    def initialize(self):
        self.options.declare('fe_model')
        self.options.declare('n_elements', types=int)
        self.options.declare('n_dof', types=int)
        self.options.declare('sK_size', types=int)

        
    def setup(self):
        fe = self.options['fe_model']
        ndof = self.options['n_dof']
        
        derivatives_k = fe.assemble_k_derivatives()
        #d_vals = 
        #d_rows = 
        #d_cols = 

        n_elements = self.options['n_elements']
        sK_size = self.options['sK_size']
                                      
        self.add_input('multipliers', shape=n_elements)
        self.add_output('stiffness_matrix', shape=(ndof,ndof))

        self.declare_partials('stiffness_matrix', 'multipliers',
                               val=derivatives_k)
                               
        # Finite difference all partials.
        #self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        fe = self.options['fe_model']
        ndof = self.options['n_dof']
        
        k = fe.assemble_K(inputs['multipliers']).toarray()
        #k_vals = k.toarray().reshape((ndof**2))
        #k = coo_matrix((val, (row, col)), shape=(ndof, ndof), dtype=np.float32).toarray()
        # Apparently, we're not able to set a sparse csc matrix as output
        outputs['stiffness_matrix'] = k
        




        
       



#TODO
# JUST WORK WITH THE REDUCED MODEL(APPLIED BOUDNARY CONDITIONS)