import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse import linalg
from scipy import linalg as nlinalg

from openmdao.api import ImplicitComponent, DirectSolver

from fem.fe_model import FEA_model


class ElasticityComp(ImplicitComponent):
    """
    """
    def initialize(self):
        self.options.declare('fe_model')
        self.options.declare('n_dof', types=int)
        self.options.declare('sK_size', types=int)
        self.options.declare('unknown_dof', types=int)


    def setup(self):
        fe = self.options['fe_model']
        bk = ~fe.bu # constrained dofs
        sK_size = self.options['sK_size']
        ndof = self.options['n_dof']
        unknown_dof = self.options['unknown_dof']
        
        self.add_input('stiffness_matrix', shape=(ndof,ndof))
        self.add_output('displacements', shape=(unknown_dof))
        

        self.declare_partials('displacements', 'displacements')


        found = np.arange(ndof)
        rows = []
        cols = []
        empty = 0
        i = 0
        for dof in bk:
            empty += ndof*dof
            if not dof:
                cols = np.hstack([cols, (found + empty)])
                rows = np.hstack([rows, np.ones(ndof)*i])
                i += 1
                empty += ndof



        self.declare_partials('displacements', 'stiffness_matrix', rows=rows, cols=cols)

        # Finite difference all partials.
        #self.declare_partials('*', '*', method='fd')

    
    def _retrieve_linear_elastic_system(self, k):
        """
        1 - Retrieves the COO sparse coordinates from the fe model and together 
        with the k_val inputs builds the CSC matrix.

        2 - Retrieve force vector and apply boundary conditions.
        """
        fe = self.options['fe_model']
        ndof = self.options['n_dof']
        
        # Build CSC Matrix from inputs + coordinates from fe model
        k = csc_matrix(k)
        #vals = k_vals
        # print('v0',k_vals.shape)
        # vals = k_vals[k_vals != 0]
        # print('v', vals.shape)
        # rows = fe.k_row
        # print(rows.shape)
        # cols = fe.k_col
        # k = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()

        
        # Retrieve fe model forces and boundary conditions
        f = fe.f
        bu = fe.bu

        # Apply boundary conditions
        Kuu = k[bu, :][:, bu]
        Fuu = f[bu]

        return Kuu, Fuu, bu

    
    def apply_nonlinear(self, inputs, outputs, residuals):
        k = self._retrieve_linear_elastic_system(inputs['stiffness_matrix'])[0]
        f = self._retrieve_linear_elastic_system(inputs['stiffness_matrix'])[1]
                
        #r1 = nlinalg.dot(k, outputs['displacements'])
        r1 = k @ outputs['displacements']
        residuals['displacements'] = r1 - f.toarray()[:,0]
        
        #k @ outputs['displacements']


    def solve_nonlinear(self, inputs, outputs):
        ndof = self.options['n_dof']
        unknown_dof = self.options['unknown_dof']

        k = self._retrieve_linear_elastic_system(inputs['stiffness_matrix'])[0]
        f = self._retrieve_linear_elastic_system(inputs['stiffness_matrix'])[1]
        bu = self._retrieve_linear_elastic_system(inputs['stiffness_matrix'])[2]

        #u = np.zeros(ndof)
        u = linalg.spsolve(k, f, use_umfpack=True)
        # Returns only the unknown displacements

        outputs['displacements'] = u#.reshape((unknown_dof,1))


    def linearize(self, inputs, outputs, partials):
        ndof = self.options['n_dof']
        unknown_dof = self.options['unknown_dof']

        k = self._retrieve_linear_elastic_system(inputs['stiffness_matrix'])[0]
        self.k = k

        partials['displacements', 'displacements'] = k


        bu = self._retrieve_linear_elastic_system(inputs['stiffness_matrix'])[2]
        u = np.zeros(ndof)
        u[bu] = outputs['displacements']

        partials['displacements', 'stiffness_matrix'] = np.tile(u, unknown_dof)


    def solve_linear(self, d_outputs, d_residuals, mode):
        k = self.k.toarray()
        if mode == 'fwd':
            d_outputs['displacements'] = nlinalg.solve(k, d_residuals['displacements'])
        elif mode == 'rev':
            d_residuals['displacements'] = nlinalg.solve(k, d_outputs['displacements']) 
        # Used to be implemented with spsolve, but as outputs can't be sparse
        # arrays, the spsolve method is ineficient.