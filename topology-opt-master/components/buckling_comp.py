from openmdao.api import ExplicitComponent

from fem.fe_model import FEA_model
from scipy.sparse import coo_matrix, csc_matrix, linalg, block_diag
import numpy as np

import time


class BucklingComp(ExplicitComponent):
    """
    """
    def initialize(self):
        self.options.declare('number_of_elements')
        self.options.declare('number_of_dof')
        self.options.declare('number_of_eigenvalues')
        self.options.declare('finite_elements_model')
        self.options.declare('minimum_x')


    def setup(self):
        nel = self.options['number_of_elements']
        neig = self.options['number_of_eigenvalues']
        fe_model = self.options['finite_elements_model']
        udof = fe_model.unknown_dof

        self.add_input('multipliers', shape=nel)
        self.add_input('states', shape=udof)
        self.add_output('critical_loads', shape=neig)

        self.declare_partials('critical_loads', 'multipliers')
        self.declare_partials('critical_loads', 'states')#, method='fd')

        ## dK(x)/dx: (constant)
        dk_dx, elements_dofs = fe_model.assemble_k_derivatives()
        print('deriv call')
        # Save constant values to use again at the compute_partials method
        self.dk_dx = dk_dx
        self.elements_dofs = elements_dofs.astype(int)

        ## Prepare dkg/du:
        dkg_du, dkg_rows, dkg_cols, element_map = fe_model.prepare_kg_u_derivatives()
        # Save values
        self.dkg_du = dkg_du
        self.dkg_rows = dkg_rows
        self.dkg_cols = dkg_cols
        self.element_map = element_map
        

        # print(dk_dx[:,:,0])
        # print(elements_dofs)

        


    def compute(self, inputs, outputs):
        fe_model = self.options['finite_elements_model']
        neig = self.options['number_of_eigenvalues']
        ndof = self.options['number_of_dof']


        # Retrieve model forces and boundary conditions
        #forces = fe_model.f
        bu = fe_model.bu
        
        ## COMPUTE STIFFNESS MATRIX [K]
        vals, rows, cols = fe_model.assemble_K(inputs['multipliers'])
        k = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()

        ## SOLVE FOR DISPLACEMENTS [u]
        kuu = k[bu, :][:, bu]
        #fuu = forces[bu]

        uu = inputs['states']#linalg.spsolve(kuu, fuu)

        ## COMPUTE GEOMETRIC STIFFNESS MATRIX
        u = np.zeros(ndof)
        u[bu] = uu
        kg = fe_model.assemble_kg(u, inputs['multipliers']).tocsc()

        ## SOLVE FOR CRITICAL BUCKLING LOADS
        kguu = kg[bu, :][:, bu]
        eigenvalues, eigv = linalg.eigsh(A=kguu, M=kuu, k=neig, return_eigenvectors=True, which='SA')

        # Normalizing eigenvectors and add constrained dofs
        eign = np.zeros((ndof, neig))
        for n in range(neig):
            #norm = np.linalg.norm(eigv[:, n])
            eign[bu,n] = eigv[:, n]#/norm
        
        # Save results to be used at compute_partials() method
        self.u = u
        self.eigenvalues = eigenvalues
        self.eigenvectors = eign
        self.bu = bu


        outputs['critical_loads'] = eigenvalues


    def compute_partials(self, inputs, partials):
        neig = self.options['number_of_eigenvalues']
        nel = self.options['number_of_elements']
        ndof = self.options['number_of_dof']
        fe_model = self.options['finite_elements_model']
        udof = fe_model.unknown_dof
        #val = self.values

        el_dofs = self.elements_dofs
        v = self.eigenvectors
        e = self.eigenvalues
        u = self.u
        bu = self.bu


        dkg_dx = fe_model.assemble_kg_derivatives(u)
        #print(dkg_du.shape)
        dk_dx = self.dk_dx

        # print('eigvec')
        # print(v[el_dofs[:,0]])

        # print(dkg_dx[:,:,0])

        # Loop runs for each element 'el' as the derivatives wrt 'x' are exclusively
        #  dependent on 'x' corresponding element
        dl_dx = np.zeros((neig, nel))
        lbd = 0
        for lbd in range(neig):
            vl = v[:,lbd]

            for n in range(nel):
                v_el = vl[el_dofs[:,n]]
                dkg_dx_el = dkg_dx[:,:,n]
                dk_dx_el = dk_dx[:,:,n]
                
                dl_dx[lbd,n] = np.transpose(v_el) @ (dkg_dx_el - e[lbd] * dk_dx_el) @ v_el


        partials['critical_loads', 'multipliers'] = dl_dx


        dkg_du = self.dkg_du
        dkg_rows = self.dkg_rows
        dkg_cols = self.dkg_cols
        element_map = self.element_map
        dkg_du_block = fe_model.assemble_dkg_du_SIMP(dkg_du, dkg_rows, dkg_cols, element_map, inputs['multipliers'])

        vl = np.zeros((ndof,1))
        dl_duu = np.zeros((neig,udof))
        for lbd in range(neig):
            
            vl[:,0] = v[:,lbd]
            # print(vl.shape)
            v_tile = np.tile(vl, (ndof,1))
            v_block = block_diag(np.transpose(np.tile(vl, ndof)))
        # print(np.transpose(np.tile(v, ndof)))
            #print(np.transpose(v_block).toarray()[:,0])

            dl_du =  (np.transpose(v_tile) @ dkg_du_block @ np.transpose(v_block))
            dl_duu[lbd,:] = dl_du[:,bu]
        #print(dl_du.shape)

        partials['critical_loads', 'states'] = dl_duu