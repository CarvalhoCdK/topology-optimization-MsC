from fem.material import Material   #pylint: disable=F0401
from fem.mesh import Mesh   #pylint: disable=F0401
from fem.builder import Builder     #pylint: disable=F0401
from fem.assembler import Assembler   #pylint: disable=F0401

from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse import linalg
import numpy as np
import time
from numba import jit, njit


class FEA_model(object):
    """
    """

   
    def __init__(self):
        self.Fe_Builder = Builder()
        self.Fe_Assembler = Assembler()

        self.ndof = None    # Number of degrees of freedom
        self.unknown_dof = None    # Number of unknown degrees of freedom
        self.k_val = None   # Global Stiffness matrix Coo values
        self.k_row = None   # Global Stiffness matrix Coo rows
        self.k_col = None   # Global Stiffness matrix Coo columns
        self.f = None    # Load vector
        self.bu = None   # Identify unknown displacement based o the boundary conditions


# Build Methods ###############################################################
    def build_fe_space(self, nelx, nely):
        
        properties = {'Nelx':nelx, 'Nely':nely}

        self.Fe_Builder.set_properties(**properties)
        self.Fe_Builder.build_fe_space()
        self.finite_elements = self.Fe_Builder.retrieve_elements()
        self.nodes = self.Fe_Builder.retrieve_nodes()
        self.ndof = 2*(nelx + 1)*(nely + 1)

        return self.finite_elements, self.nodes

    
    def set_material(self, material_properties):

        self.material = Material(**material_properties)
        self.constitutive_model = self.material.get_elasticity_matrix()     


    def set_loads(self, f):

        self.f = csc_matrix(f)
    

    def set_boundary_conditions(self, bk):  

        self.bu = ~bk # defining unknown DOFs
        self.fu = self.f[self.bu]
        self.unknown_dof = self.fu.shape[0]
                
    
# Assemble methods ############################################################
    def assemble_K(self, artificial_densities='No SIMP'):
        
        constitutive_model = self.constitutive_model
        finite_elements = self.finite_elements
        #ndof = self.ndof
        
        (vals, rows, cols) = self.Fe_Assembler.assemble_stiffness_matrix(finite_elements,
                                                             constitutive_model,
                                                             densities=artificial_densities)
        
        self.k_val = vals
        self.k_row = rows
        self.k_col = cols
        #self.k = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof))
                
        return vals, rows, cols

    
    def assemble_k_derivatives(self, x_min=1e-6):

        model = self.constitutive_model
        elements = self.finite_elements
        ndof = self.ndof
        assembler = self.Fe_Assembler
        
        k_derivs, k_deriv_dofs = assembler.assemble_k_derivatives(elements,
                                                                  model,
                                                                  ndof,
                                                                  x_min)


        # k_derivs = self.Fe_Assembler.assemble_stiffness_matrix_derivatives(elements,
        #                                                                    model,
        #                                                                    ndof)

        return k_derivs, k_deriv_dofs


    def assemble_kg_derivatives(self, nodal_displacements):

        model = self.constitutive_model
        elements = self.finite_elements
        #ndof = self.ndof
        assembler = self.Fe_Assembler
        
        kg_derivs= assembler.assemble_kg_derivatives(elements,
                                                     model,
                                                     nodal_displacements)

        return kg_derivs


    def assemble_kg(self, nodal_displacements, densities):

        constitutive_model = self.constitutive_model
        finite_elements = self.finite_elements
        ndof = self.ndof

        vals, rows, cols = self.Fe_Assembler.assemble_geometric_stiffness_matrix(finite_elements,
                                                              constitutive_model,
                                                              nodal_displacements,
                                                              densities)


        self.kg = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof))
        
        return self.kg


    def assemble_ku_derivatives(self, nodal_displacements, only_data=False):
        constitutive_model = self.constitutive_model
        finite_elements = self.finite_elements
        assembler = self.Fe_Assembler
        ndof = self.ndof
        displacements = nodal_displacements

        dku = assembler.assemble_ku_derivatives(ndof,
                                                finite_elements,
                                                constitutive_model,
                                                displacements,
                                                x_min=1e-6,
                                                only_data=False)
        
        
        return dku



    def prepare_kg_u_derivatives(self):
        constitutive_model = self.constitutive_model
        finite_elements = self.finite_elements
        assembler = self.Fe_Assembler
        ndof = self.ndof

        dkg_du, dkg_rows, dkg_cols, element_map = assembler.prepare_kg_u_derivatives(finite_elements,
                                                   constitutive_model,
                                                   ndof)
        
        return dkg_du, dkg_rows, dkg_cols, element_map


    def assemble_dkg_du_SIMP(self, dkg_du, dkg_rows, dkg_cols, element_map, densities):
        assembler = self.Fe_Assembler

        dkg_du_block = assembler.assemble_dkg_du_SIMP(dkg_du, dkg_rows, dkg_cols, element_map, densities)
        return dkg_du_block

# Solvers ######################################################################
    def solve_elastic(self):
        ndof = self.ndof
        vals = self.k_val
        rows = self.k_row
        cols = self.k_col
        #vals, rows, cols = self.assemble_K()

        self.k = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
        k = self.k
        f = self.f
        bu = self.bu

        # sub-matrices corresponding to unknown DOFs
        Kuu = k[bu, :][:, bu]
        Fuu = f[bu,:]

        u = np.zeros(self.ndof)
        u[bu] = linalg.spsolve(Kuu, Fuu)

        return u

    def solve_stability(self, eigenvectors=False):

        k = self.k
        kg = self.kg.tocsc()
        bu = self.bu

        Kuu = k[bu, :][:, bu]
        Kguu = kg[bu, :][:, bu]
        if not eigenvectors:
            eigenvalues= linalg.eigsh(A=Kguu,M=Kuu, k=5, return_eigenvectors=eigenvectors, tol=8)
            return -1/eigenvalues
        else:
            eigenvalues, vec = linalg.eigsh(A=Kguu,M=Kuu, k=5)
            return -1/eigenvalues, vec
        ## ERROR
        # A = Kguu @ linalg.inv(Kuu)
        # eigenvalues= linalg.eigsh(A=A, k=5, return_eigenvectors=False, tol=2)
        # Formato muito mais lento ^

        return -1/eigenvalues
   


 



## CLASS TEST

# fe = FEA_model()
# fe.build_set_mesh()
# fe.assemble_K()
# fe.set_loads()
# k, f = fe.set_boundary_conditions()
# u = fe.solve_elastic(k,f)

#val, row, col = fe.assemble_K()
#ndof = int(max(col)+1)  # +1 to convert array max "position" to "size"
#k = coo_matrix((val, (row, col)), shape=(ndof, ndof)).tocsc()

# print(fe.Test_var)

# FEA_model.Test_var = 2
# print(fe.Test_var)
# print('EoT')

## END OF CLASS TEST
