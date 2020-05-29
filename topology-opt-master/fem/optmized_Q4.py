## Dependencies
import numpy as np
from numba import njit, jit

import time
##

class OptElementQ4(object):
    """
    Q4 finite element.

    0------1
    |      |
    |      |
    2------3

    """
    regular_mesh = True
    ke_regular = None

    integration_points = np.array([-0.5773502691896257, 0.5773502691896257])
    thickness = 0.005

    __slots__ = ['element_nodes', 'nodal_dofs', 'B_int_points', 'G_int_points','ke_rows', 'ke_cols']


    def __init__(self, element_nodes):

        self.element_nodes = element_nodes
        ke_rows, ke_cols = self._dof_coordinates(element_nodes)

        self.ke_rows = ke_rows
        self.ke_cols = ke_cols



    def _dof_coordinates(self, element_nodes):
        """
        Identify the global coordinates for each degree of freedom in
        an element.

        Returns:
            rows_ke: ndarray[64,1]
            cols_ke: ndarray[64,1]
        """
       
        n_nodes = len(element_nodes)
        dof_map = np.zeros((n_nodes, 2)).astype(int)

        for n in range(n_nodes):
            dof_map[n, 0] = 2*element_nodes[n]
            dof_map[n, 1] = 2*element_nodes[n]+1

        self.nodal_dofs = dof_map.flatten()
        
        [i, j] = np.meshgrid(self.nodal_dofs, self.nodal_dofs)

        rows_ke = i.flatten()
        cols_ke = j.flatten()
        
        return rows_ke, cols_ke


    @staticmethod
    def _stress_strain_matrix(ksi, eta):
        """
        Return the stress-strain matrix for a Q4 square element with unity side.

        """
        nodes = 4
        inv_J = 2   # Inverse Jacobian matrix  = inv_J * [[1, 0],[-1, 0]]

        shape_derivatives = np.zeros((2,4))

        shape_derivatives[0,0] = -eta-1   #- 1 + eta
        shape_derivatives[0,1] =  eta+1   # 1 - eta
        shape_derivatives[0,2] =  eta-1   # 1 + eta
        shape_derivatives[0,3] = -eta+1   #- 1 - eta
        shape_derivatives[1,0] =  ksi-1 #-ksi+1   
        shape_derivatives[1,1] = -ksi-1# ksi+1  
        shape_derivatives[1,2] = -ksi+1# ksi-1   
        shape_derivatives[1,3] =  ksi+1#-ksi-1 

        # Convert local derivatives (x, y) to global (ksi, eta)
        shape_derivatives *= inv_J/4

        B = np.zeros((3, 2*nodes))
        for i in range(nodes):
            col1 = 2*i
            col2 = col1+1

            B[0,col1] = shape_derivatives[0,i]
            B[1,col2] = shape_derivatives[1,i]
            B[2,col1] = shape_derivatives[1,i]
            B[2,col2] = shape_derivatives[0,i]
          
        
        G = np.zeros((4,2*nodes))
        for i in range(nodes):
            col1 = 2*i
            col2 = col1+1

            G[0,col1] = shape_derivatives[0,i]
            G[1,col1] = shape_derivatives[1,i]
            G[2,col2] = shape_derivatives[0,i]
            G[3,col2] = shape_derivatives[1,i]

        
        return B, G


    @classmethod
    def shape_derivatives_at_int_points(cls):
        """
        
        Calculate, at the gauss points for integration, the matrix B and G
        containing shape derivatives.
        In a regular mesh, stores those values as class atributes

        """
        int_point = cls.integration_points

        # Integration point 1
        ksi = int_point[0]
        eta = int_point[0]
        B1, G1 = cls._stress_strain_matrix(ksi, eta)

        # Integration point 2
        ksi = int_point[0]
        eta = int_point[1]
        B2, G2 = cls._stress_strain_matrix(ksi, eta)
        
        # Integration point 3
        ksi = int_point[1]
        eta = int_point[0]
        B3, G3 = cls._stress_strain_matrix(ksi, eta)

        # Integration point 4
        ksi = int_point[1]
        eta = int_point[1]
        B4, G4 = cls._stress_strain_matrix(ksi, eta)

        B = {'B1':B1, 'B2':B2, 'B3':B3, 'B4':B4}
        G = {'G1':G1, 'G2':G2, 'G3':G3, 'G4':G4}

        cls.B_int_points = B
        cls.G_int_points = G

        
    @classmethod
    def _store_stiffness_matrix(cls, ke_vals, ke):
        
        if cls.regular_mesh:
            cls.ke_regular = ke_vals
            cls.ke_r = ke

    
    def get_stiffness_matrix(self,
                             constitutive_model,
                             derivatives=False):
        """
        """
        np.set_printoptions(precision=4)
        thickness = self.thickness
        #element_nodes = self.element_nodes
        
        # For a regular mesh, the element just fetches the stiffness matrix values 
        # calculated for the first element.
        if self.regular_mesh and self.ke_regular is not None:
            #ke_rows, ke_cols = self._dof_coordinates(element_nodes)
            #print("Regular fetch")

            if derivatives:
                return self.ke_r, self.nodal_dofs
            else:
                return self.ke_regular, self.ke_rows, self.ke_cols


        # Calculate shape derivatives at gauss points
        self.shape_derivatives_at_int_points()
        
        C = constitutive_model
        J = 0.25#4 # Jacobian
        n_dof = 8

        ke = np.zeros((n_dof, n_dof))

        # Integration point 1
        B = self.B_int_points['B1']
        ke1 = np.transpose(B) @ C @ B
                
        # Integration point 2
        B = self.B_int_points['B2']
        ke2 = np.transpose(B) @ C @ B
               
        # Integration point 3
        B = self.B_int_points['B3']
        ke3 = np.transpose(B) @ C @ B
        
        # Integration point 4
        B = self.B_int_points['B4']
        ke4 = np.transpose(B) @ C @ B
                
        ke = J*(ke1 + ke2 + ke3 + ke4)*thickness
        
        # Build 1D arrays for sparse ke
        ke_vals = ke.flatten()
        ke_rows = self.ke_rows
        ke_cols = self.ke_cols
        #ke_rows, ke_cols = self._dof_coordinates(element_nodes)


        if self.regular_mesh and self.ke_regular is None:
            self._store_stiffness_matrix(ke_vals, ke)

        if derivatives:
            return ke, self.nodal_dofs
        else:
            return ke_vals, ke_rows, ke_cols

    
    def _get_stresses(self, constitutive_model, nodal_displacements):
        """
        Calculate stresses at the gauss points
        """

        C = constitutive_model
        u = nodal_displacements
       
        # Integration point 1
        B = self.B_int_points['B1']
        s1 = C @ B @ u

        # Integration point 2
        B = self.B_int_points['B2']
        s2 = C @ B @ u
                       
        # Integration point 3
        B = self.B_int_points['B3']
        s3 = C @ B @ u
                
        # Integration point 4
        B = self.B_int_points['B4']
        s4 = C @ B @ u
        #print(s4)
        
        return np.array([s1, s2, s3, s4])


    def get_geometric_stiffness_matrix(self,
                                       constitutive_model,
                                       nodal_displacements,
                                       derivatives=False):
        """
        """
        stress = self._get_stresses(constitutive_model, nodal_displacements)

        #n_dof = 8
        J = 0.25#4 # Jacobian
        thickness = self.thickness

        # Reshape stress arrays
        S = dict()
        int_point_id = 1
        for int_point_stress in stress:

            stress = np.zeros((4,4))

            stress[0,0] = int_point_stress[0]
            stress[0,1] = int_point_stress[2]
            stress[1,0] = int_point_stress[2]
            stress[1,1] = int_point_stress[1]
        
            stress[2,2] = int_point_stress[0]
            stress[2,3] = int_point_stress[2]
            stress[3,2] = int_point_stress[2]
            stress[3,3] = int_point_stress[1]

            key = 'S' + str(int_point_id)
            S[key] = stress
            int_point_id += 1

            #print('key:', key, 'S:', S[key])

        # Integration point 1
        G = self.G_int_points['G1']
        kg1 = np.transpose(G) @ S['S1'] @ G
        
        # Integration point 2
        G = self.G_int_points['G2']
        kg2 = np.transpose(G) @ S['S2'] @ G

        # Integration point 3
        G = self.G_int_points['G3']
        kg3 = np.transpose(G) @ S['S3'] @ G

        # Integration point 4
        G = self.G_int_points['G4']
        kg4 = np.transpose(G) @ S['S4'] @ G

        kg = J*(kg1 + kg2 + kg3 + kg4)*thickness

        # Build 1D arrays for sparse ke
        if derivatives:
            return kg
        else:
            kg_vals = kg.flatten()
            kg_rows = self.ke_rows
            kg_cols = self.ke_cols
            #kg_rows, kg_cols = self._dof_coordinates(element_nodes)
            return kg_vals, kg_rows, kg_cols


    
    def get_geometric_matrix_u_derivs(self, constitutive_model):
        """
        """
        element_dofs = self.nodal_dofs

        # Start dictionaries
        dkge_vals = dict()
        #dkge_rows = dict()
        #dkge_cols = dict()

        for i in np.arange(8):
            dui = np.zeros(8)
            dui[i] = 1

            ds_dui = self._get_stresses(constitutive_model, dui)

            #n_dof = 8
            J = 0.25    # Jacobian
            thickness = self.thickness

            # Reshape stress arrays for sigma[3,1] to S[4,4]
            S = dict()
            int_point_id = 1
            for int_point_stress in ds_dui:

                s = np.zeros((4,4))

                s[0,0] = int_point_stress[0]
                s[0,1] = int_point_stress[2]
                s[1,0] = int_point_stress[2]
                s[1,1] = int_point_stress[1]
            
                s[2,2] = int_point_stress[0]
                s[2,3] = int_point_stress[2]
                s[3,2] = int_point_stress[2]
                s[3,3] = int_point_stress[1]

                key = 'S' + str(int_point_id)
                S[key] = s
                int_point_id += 1

            # Integration point 1
            G = self.G_int_points['G1']
            kg1 = np.transpose(G) @ S['S1'] @ G
            
            # Integration point 2
            G = self.G_int_points['G2']
            kg2 = np.transpose(G) @ S['S2'] @ G

            # Integration point 3
            G = self.G_int_points['G3']
            kg3 = np.transpose(G) @ S['S3'] @ G

            # Integration point 4
            G = self.G_int_points['G4']
            kg4 = np.transpose(G) @ S['S4'] @ G

            kg = J * (kg1 + kg2 + kg3 + kg4) * thickness
            
            kg_vals = kg.flatten()
            
            # dKge(u)/du dictionary, one entry[8,8 or 64 flat] for each dof of the
            # element. The entry number is based on the global dof naming convention
            dkge_vals[element_dofs[i]] = kg_vals
            
        dkg_rows = self.ke_rows
        dkg_cols = self.ke_cols

        return dkge_vals, dkg_rows, dkg_cols


















        





        
## CLASS TEST
#from material import Material
# nodes = [0,1,2,3]
# el = OptElementQ4(nodes)



# material_properties = {'Ex': 1,    # GPa
#                        'Ey': 1,    # GPa
#                        'NUxy': 0.3}
# pla = Material(**material_properties)
# c = pla.get_elasticity_matrix()

# ke, row, col = el.get_stiffness_matrix(c)
# print("EoT")



