from scipy.sparse import coo_matrix, csc_matrix, linalg

from openmdao.api import ImplicitComponent, ExplicitComponent, DirectSolver



class StabilityComp(ExplicitComponent):
    """
    """
    def initialize(self):
        self.options.declare('fe_model')
        self.options.declare('n_dof', types=int)
        self.options.declare('sK_size', types=int)
        self.options.declare('unknown_dof', types=int)
        self.options.declare('n_eigenvalues', types=int)


    def setup(self):
        ndof = self.options['n_dof']
        n_eig = self.options['n_eigenvalues']

        self.add_input('stiffness_matrix', shape=(ndof,ndof))
        self.add_input('geometric_matrix', shape=(ndof,ndof))

        self.add_output('critical_loads', shape=(n_eig))

        self.declare_partials('critical_loads', 'stiffness_matrix', method='fd')
        self.declare_partials('critical_loads', 'geometric_matrix', method='fd')


    def compute(self, inputs, outputs):
        fe = self.options['fe_model']
        n_eig = self.options['n_eigenvalues']
        
        k = csc_matrix(inputs['stiffness_matrix'])
        kg = csc_matrix(inputs['geometric_matrix'])

        bu = fe.bu
        Kuu = k[bu, :][:, bu]
        Kguu = kg[bu, :][:, bu]
      
        eigenvalues = linalg.eigsh(A=Kguu,M=Kuu, k=n_eig, return_eigenvectors=False)
        outputs['critical_loads'] = -1/eigenvalues

    # def compute_partials(self, inputs, partials):
    #     fe = self.options['fe_model']
    #     n_eig = self.options['n_eigenvalues']
        
    #     k = csc_matrix(inputs['stiffness_matrix'])
    #     kg = csc_matrix(inputs['geometric_matrix'])

    #     bu = fe.bu
    #     Kuu = k[bu, :][:, bu]
    #     Kguu = kg[bu, :][:, bu]

    #     (e, v) = linalg.eigsh(A=Kguu,M=Kuu, k=n_eig, return_eigenvectors=False, tol=5)








## IMPLICIT VERSION
# class StabilityComp(ImplicitComponent):
#     """
#     """
#     def initialize(self):
#         self.options.declare('fe_model')
#         self.options.declare('n_dof', types=int)
#         self.options.declare('sK_size', types=int)
#         self.options.declare('unknown_dof', types=int)
#         self.options.declare('n_eigenvalues', types=int)


#     def setup(self):
#         n_eigenvalues = self.options['n_eigenvalues']

#         self.add_input('stiffness_matrix', shape=(ndof,ndof))
#         self.add_input('geometric_matrix', shape=(ndof,ndof))

#         self.add_output('critical_load', shape=(n_eigenvalues))


#     def apply_linear(self, inputs, outputs, residuals):
        
#         k = csc_matrix(inputs['stiffness_matrix'])
#         kg = csc_matrix(inputs['geometric_matrix'])

#         residuals['critical_load'] = 
