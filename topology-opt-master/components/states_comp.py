from openmdao.api import ImplicitComponent

from scipy.sparse import coo_matrix, csc_matrix, linalg
import numpy as np




class States_comp(ImplicitComponent):
    """
    """
    def initialize(self):
        self.options.declare('number_of_elements')
        self.options.declare('number_of_dof')
        self.options.declare('finite_elements_model')
        self.options.declare('minimum_x')


    def setup(self):
        nel = self.options['number_of_elements']
        ndof = self.options['number_of_dof']
        fe_model = self.options['finite_elements_model']
        udof = fe_model.unknown_dof
        bu = fe_model.bu
        #bk = ~bu

        self.add_input('multipliers', shape=(nel))

        self.add_output('states', shape=(udof))

        # Building sparse indices for the partial dR(u)/du
        vals, rows, cols = fe_model.assemble_K(np.ones(nel))
        # k = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
        # kuu = k[bu, :][:, bu].tocoo()
        # rows = kuu.row
        # cols = kuu.col
        self.declare_partials('states', 'states')# rows=rows, cols=cols)

        # Build sparse indices for the partial dR(u)/dx
        # dku = fe_model.assemble_ku_derivatives(np.ones(ndof)).tocsc()
        # dku_uu = dku[bu, :].tocoo()
        # rows = dku_uu.row
        # cols = dku_uu.col
        self.declare_partials('states', 'multipliers') #, rows=rows, cols=cols)



    def retrieve_k_matrix(self, inputs):
        fe_model = self.options['finite_elements_model']
        ndof = self.options['number_of_dof']

        # Retrieve model forces and boundary conditions
        forces = fe_model.f
        bu = fe_model.bu
        
        ## COMPUTE STIFFNESS MATRIX [K]
        vals, rows, cols = fe_model.assemble_K(inputs['multipliers'])
        k = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()

        ## SOLVE FOR DISPLACEMENTS [u]
        kuu = k[bu, :][:, bu]
        fuu = forces[bu]

        return kuu, fuu


    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        R(u) = KU - F
        """
        k, f = self.retrieve_k_matrix(inputs)

        residuals['states'] = k.dot(outputs['states']) - f.toarray()[:,0]

    
    def solve_nonlinear(self, inputs, outputs):
        """
        U = K^(-1)F
        """
        k, f = self.retrieve_k_matrix(inputs)

        outputs['states'] = linalg.spsolve(k, f)

    
    def linearize(self, inputs, outputs, partials):
        fe_model = self.options['finite_elements_model']
        ndof = self.options['number_of_dof']
        bu = fe_model.bu

        # Prepare partial dR(u)/du
        k = self.retrieve_k_matrix(inputs)[0]
        self.k = k
        #k_vals = k.tocoo().data

        partials['states', 'states'] = k

        # Prepare partial dR(u)/dx
        u = np.zeros(ndof)
        u[bu] = outputs['states']

        dku = fe_model.assemble_ku_derivatives(u).tocsc()
        dku_uu = dku[bu, :]
        #dku_vals = dku_uu.tocoo().data

        
        partials['states', 'multipliers'] = dku_uu
        


    def solve_linear(self, d_outputs, d_residuals, mode):
        k = self.k
        if mode == 'fwd':
            d_outputs['states'] = linalg.spsolve(k, d_residuals['states'])
        
        elif mode == 'rev':
            d_residuals['states'] = linalg.spsolve(k, d_outputs['states'])

