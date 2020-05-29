from openmdao.api import Group, IndepVarComp, NewtonSolver, DirectSolver
#C:\Users\LABOSS-1\Documents\Luiz\MsC Codes\Topology\topology-opt\components\penalization_comp.py

from components.penalization_comp import PenalizationComp
from components.filter_comp import DensityFilterComp
from components.stiffness_matrix_comp import StiffnessMatrixComp
from components.geometric_matrix import GeometricMatrixComp
from components.elasticity_comp import ElasticityComp
from components.stability_comp import StabilityComp
from components.compliance_comp import ComplianceComp
from components.volume_comp import VolumeComp

import numpy as np
from fem.fe_model import FEA_model


class SimpGroup(Group):
    """
    """

    def initialize(self):
        # FE model: domain, loads, boundary conditions, material.
        self.options.declare('fe_model')
        # SIMP Topology optimization parameters
        self.options.declare('num_ele_x', types=int)
        self.options.declare('num_ele_y', types=int)
        self.options.declare('penal_factor', types=(int, float))
        self.options.declare('volume_fraction', types=float)
        self.options.declare('filter_radius', types=float)


    def setup(self):
        fe_model = self.options['fe_model']

        nelx = self.options['num_ele_x']
        nely = self.options['num_ele_y']
        penal = self.options['penal_factor']
        volume_fraction = self.options['volume_fraction']
        radius = self.options['filter_radius']



        # Check FEA assembly and evaluate dimensions of the sparse matrix
        n_elements = nelx*nely
        n_dof = 2*(nelx+1)*(nely+1)
        k_vals = fe_model.k_val#assemble_K(np.ones(n_elements)).flatten()
        sK_size = len(k_vals)   # Sparse arrays length under BC.
        unknown_dof = fe_model.unknown_dof
        n_eigenvalues = 1

        # Setup design variables ['densities']
        comp = IndepVarComp()
        comp.add_output('densities', val=1.0, shape=n_elements)
        comp.add_design_var('densities', lower=0.01, upper=1.0)
        self.add_subsystem('input_comp', comp)
        # With Filter
        self.connect('input_comp.densities',
                     'filter_comp.densities')
        # No Filter
        # self.connect('input_comp.densities',
        #              'volume_comp.densities')

        # Density Filter
        comp = DensityFilterComp(num_ele_x=nelx, num_ele_y=nely, radius=radius)
        self.add_subsystem('filter_comp', comp)
        self.connect('filter_comp.densities_f',
                     'penalization_comp.densities')
        self.connect('filter_comp.densities_f',
                     'volume_comp.densities')
        

        # Penalization
        comp = PenalizationComp(penal=penal, n_elements=n_elements)
        self.add_subsystem('penalization_comp', comp)
        self.connect('penalization_comp.multipliers',
                     'stiffness_matrix_comp.multipliers')

        # Stiffness Matrix
        comp = StiffnessMatrixComp(fe_model=fe_model,
                                   n_elements=n_elements,
                                   n_dof=n_dof,
                                   sK_size=sK_size)
        self.add_subsystem('stiffness_matrix_comp', comp)
        self.connect('stiffness_matrix_comp.stiffness_matrix',
                     'elasticity_comp.stiffness_matrix')
        self.connect('stiffness_matrix_comp.stiffness_matrix',
                     'stability_comp.stiffness_matrix')             

        # Elasticity 
        comp = ElasticityComp(fe_model=fe_model,
                               n_dof=n_dof,
                               sK_size=sK_size,
                               unknown_dof=unknown_dof)
        self.add_subsystem('elasticity_comp', comp)
        self.connect('elasticity_comp.displacements',
                     'compliance_comp.displacements')
        self.connect('elasticity_comp.displacements',
                     'geometric_matrix_comp.displacements')

        # Geometric Stiffness Matrix
        comp = GeometricMatrixComp(fe_model=fe_model,
                               n_dof=n_dof,
                               unknown_dof=unknown_dof)
        self.add_subsystem('geometric_matrix_comp', comp)
        self.connect('geometric_matrix_comp.geometric_matrix',
                     'stability_comp.geometric_matrix')
        

        # Stability
        comp = StabilityComp(fe_model=fe_model,
                             n_dof=n_dof,
                             sK_size=sK_size,
                             unknown_dof=unknown_dof,
                             n_eigenvalues=n_eigenvalues)
        self.add_subsystem('stability_comp', comp)

        

        # # Compliance
        comp = ComplianceComp(fe_model=fe_model,
                              ndof=n_dof,
                              unknown_dof=unknown_dof)
        self.add_subsystem('compliance_comp', comp)

        # Volume
        comp = VolumeComp(nelements=n_elements)
        self.add_subsystem('volume_comp', comp)

        # Stablity constraint / aggregation function


        # Design variables
        self.add_design_var('input_comp.densities', lower=1e-2, upper=1)

        # Objective
        self.add_objective('compliance_comp.compliance')
        
        # Constraints:
        self.add_constraint('volume_comp.volume', upper=volume_fraction, linear=True)

        self.add_constraint('stability_comp.critical_loads', upper=51)
        

        # volume constraint

        # self.nonlinear_solver = NewtonSolver()
        #self.linear_solver = DirectSolver()
        #self.options['assembled_jac_type'] = 'csc'
        #self.linear_solver = DirectSolver(assemble_jac=True)

print('EoF')