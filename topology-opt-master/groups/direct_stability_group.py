from openmdao.api import Group, IndepVarComp

from components.penalization_comp import PenalizationComp
from components.filter_comp import DensityFilterComp
from components.buckling_comp import BucklingComp
from components.states_comp import States_comp
from components.compliance_comp import ComplianceComp
from components.volume_comp import VolumeComp
from components.buckling_constraint_comp import BucklingConstraint
from components.aggregation_constraint_comp import AggregationConstraint

import numpy as np


class DirectStabilityGroup(Group):
    """
    OpenMDAO group develop to adapt linear buckling constraints to a topology
    optimization model. A direct approach, instead of modular components of K 
    and Kg, is used supress the need of an Implicit component with two outputs
    and teh associated excess need of derivative calculations. This new group
    should conect 'multipliers' to the 'stability_comp' and directly output the 
    buckling eigenvalues.
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
        self.options.declare('max_buckling_load', types=float)
        self.options.declare('initial_densities', types=(float, np.ndarray))
        self.options.declare('data_recorder', default=False)


    def setup(self):
        fe_model = self.options['fe_model']
        nelx = self.options['num_ele_x']
        nely = self.options['num_ele_y']
        penal = self.options['penal_factor']
        volume_fraction = self.options['volume_fraction']
        radius = self.options['filter_radius']
        max_buckling_load = self.options['max_buckling_load']
        initial_x = self.options['initial_densities']
        data_recorder = self.options['data_recorder']

        # Check FEA assembly and evaluate dimensions of the sparse matrix
        n_elements = nelx*nely
        n_dof = 2*(nelx+1)*(nely+1)
        unknown_dof = fe_model.unknown_dof
        
        # Inside default parameters
        n_eigenvalues = 5
        eigenvalue_gap = 1.01
        #aggregation_parameter = 30.0
        minimum_x = 1e-6

        # Setup design variables ['densities']
        comp = IndepVarComp()
        comp.add_output('densities', val=initial_x, shape=n_elements)
        comp.add_design_var('densities', lower=0.0, upper=1.0)
        self.add_subsystem('input_comp', comp)
        self.connect('input_comp.densities',
                     'filter_comp.densities')
                     
        # self.connect('input_comp.densities',
        #              'penalization_comp.densities')
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
                     'buckling_comp.multipliers')
        self.connect('penalization_comp.multipliers',
                     'states_comp.multipliers')

        ## STATES COMP
        comp = States_comp(number_of_elements=n_elements,
                           number_of_dof = n_dof,
                           finite_elements_model=fe_model,
                           minimum_x=minimum_x)
        self.add_subsystem('states_comp', comp)
        self.connect('states_comp.states',
                     'compliance_comp.displacements')
        self.connect('states_comp.states',
                     'buckling_comp.states')            

        ## BUCKLING COMP
        comp = BucklingComp(number_of_dof = n_dof,
                            number_of_elements=n_elements,
                            number_of_eigenvalues=n_eigenvalues,
                            finite_elements_model=fe_model,
                            minimum_x=minimum_x)
        self.add_subsystem('buckling_comp', comp)
        self.connect('buckling_comp.critical_loads',
                     'buckling_constraint_comp.eigenvalues')


        # Compliance
        comp = ComplianceComp(fe_model=fe_model,
                              ndof=n_dof,
                              unknown_dof=unknown_dof,
                              data_recorder=data_recorder)
        self.add_subsystem('compliance_comp', comp)

        # Volume
        comp = VolumeComp(nelements=n_elements,
                          data_recorder=data_recorder)
        self.add_subsystem('volume_comp', comp)

        # Buckling Constraint Comp - Direct engenvalues
        comp = BucklingConstraint(eigenvalue_gap=eigenvalue_gap,
                                  max_buckling_load=max_buckling_load,
                                  number_of_eigenvalues=n_eigenvalues,
                                  data_recorder=data_recorder)
        self.add_subsystem('buckling_constraint_comp', comp)

        # Buckling Constraint Comp - Aggregation
        # comp = AggregationConstraint(aggregation_parameter=aggregation_parameter,
        #                              max_buckling_load=max_buckling_load,
        #                              number_of_eigenvalues=n_eigenvalues)
        # self.add_subsystem('aggregation_constraint_comp', comp)


        # Design variables
        #self.add_design_var('input_comp.densities', upper=1)

        # Objective
        #self.add_objective('volume_comp.volume')
        self.add_objective('compliance_comp.compliance')
        
        # Constraints:
        self.add_constraint('volume_comp.volume', upper=volume_fraction, linear=True)

        #self.add_constraint('aggregation_constraint_comp.residuals', lower=0)
        self.add_constraint('buckling_constraint_comp.residuals', lower=0)