import numpy as np
from openmdao.api import ExplicitComponent
from fem.fe_model import FEA_model


class ComplianceComp(ExplicitComponent):
    """
    Takes only displacements from unknown degrees of freedom, ence in this fea
    model, the other DOFs are set to u = 0.
    """

    def initialize(self):
        self.options.declare('fe_model')
        self.options.declare('ndof', types=int)
        self.options.declare('unknown_dof', types=int)
        self.options.declare('data_recorder', default=False)


    def setup(self):
        unknown_dof = self.options['unknown_dof']
        
        self.add_input('displacements', shape=unknown_dof)
        self.add_output('compliance')
        
        self.declare_partials('compliance', 'displacements')
 

    def compute(self, inputs, outputs):
        forces = self.options['fe_model'].fu
        

        compliance = inputs['displacements'] @ forces
        self.compliance = compliance

 
        outputs['compliance'] = compliance

        

    def compute_partials(self, inputs, partials):
        forces = self.options['fe_model'].fu.transpose()

        data_recorder = self.options['data_recorder']
        ## Recording information
        if data_recorder != False:
            data_recorder.get_compliance(self.compliance)
        #u_dof = self.options['unknown_dof']

        partials['compliance', 'displacements'] = forces