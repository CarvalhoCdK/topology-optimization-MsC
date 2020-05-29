## Dependencies
import numpy as np
import time
from scipy.sparse import coo_matrix, hstack, block_diag, issparse

from fem.element import Q4_Element 
from numba import jit, njit 

##


class Assembler(object):
    """
    Aggregate methods for building the global matrices for the FEM model.

    """

    def __init__(self):
        pass

    
    @staticmethod
    def assemble_stiffness_matrix(finite_elements,
                                  constitutive_model,
                                  densities='NO SIMP',
                                  x_min=1e-6):
        """
        Compute the elements stiffness matrix (ke) and assemble the global matrix (k)
        already applying the element densities for topology optimization.
        
        Consider:
        Q4 element

        K(x) = (x_min + x(1 - x_min)) K0
        dK(x)/dx = (1 - x_min)K0

        """
        #print("Assemble call")
        
        nel = len(finite_elements)
        # No topology optimization 
        if type(densities) == str:
            densities = np.ones(nel)
        
        # k_vals = np.array([])
        # k_rows = np.array([])
        # k_cols = np.array([])

        # elem = np.arange(nel)  
        # for el in np.nditer(elem):
        #     ke_vals, ke_rows, ke_cols = finite_elements[el].get_stiffness_matrix(constitutive_model)
            
        #     x = x_min + densities[el]* (1 - x_min)
            
        #     k_vals = np.append(k_vals, ke_vals*x)#k_vals.extend(ke_vals * densities[el])
        #     k_rows = np.append(k_rows, ke_rows)#k_rows.extend(ke_rows)
        #     k_cols = np.append(k_cols, ke_cols)#k_cols.extend(ke_cols)

        ########################################################################
        #### Specific fast calculation for the Q4 element
        size = nel*64
        k_vals = np.zeros(size)
        k_rows = np.zeros(size)
        k_cols = np.zeros(size)

        elem = np.arange(nel)
        start = 0
        finish = 64
        for el in np.nditer(elem):
            ke_vals, ke_rows, ke_cols = finite_elements[el].get_stiffness_matrix(constitutive_model)
            
            x = x_min + densities[el]* (1 - x_min)
            
            k_vals[start:finish] = ke_vals*x#k_vals.extend(ke_vals * densities[el])
            k_rows[start:finish] = ke_rows#k_rows.extend(ke_rows)
            k_cols[start:finish] = ke_cols#k_cols.extend(ke_cols)

            start += 64
            finish += 64

        #k_vals = np.array(k_vals)
        #k_rows = np.array(k_rows)
        #k_cols = np.array(k_cols)
        #ndof = int(max(k_cols)+1)  # +1 to convert array max "position" to "size"
# elem = np.arange(10)
# s = 0
# f = 4
# p = 1
# a=np.zeros(10*4)
# for el in np.nditer(elem):
#     b= np.ones(4)*p
#     p+=1
#     a[s:f] = b

#     s +=4
#     f +=4




        #k = coo_matrix((k_vals, (k_rows, k_cols)), shape=(ndof, ndof)).tocsc()
        return k_vals, k_rows, k_cols


    @staticmethod
    def assemble_stiffness_matrix_derivatives(finite_elements,
                                              constitutive_model,
                                              ndof):
        """
        Consider:
        Q4 element

        K(x) = (x_min + x(1 - x_min)) K0
        dK(x)/dx = (1 - x_min)K0

        """
        nel = len(finite_elements)

        #dk_vals = []
        #dk_rows = []
        #dk_cols = []
        derivatives_k = []

        for el in range(nel):
            ke_vals, ke_rows, ke_cols = finite_elements[el].get_stiffness_matrix(constitutive_model)
            dke = coo_matrix((ke_vals, (ke_rows, ke_cols)), shape=(ndof, ndof))

            dke_line = dke.reshape((int(ndof**2),1))#.tocsc()
            #dke_line = dke_line[dke_line != 0]

            if el == 0:
                derivatives_k = dke_line
            else:
                derivatives_k = hstack([derivatives_k, dke_line])    
            # separar a matriz d ederivadas de cada elemento, fllatten, e concatenar

            #dk_vals.extend(dk_line.data)
            #dk_rows.extend(dk_line.row)
            #dk_cols.extend(dk_line.col)

        # dk_vals = np.array(dk_vals)
        # dk_rows = np.array(dk_rows)
        # dk_cols = np.array(dk_cols)

        return derivatives_k


    @staticmethod
    def assemble_geometric_stiffness_matrix(finite_elements, constitutive_model, nodal_displacements, densities):
        """
        Compute the elements geometric stiffness matrix (kge) and assemble the global 
        matrix (kg) already applying the element densities for topology optimization.
        """

        nel = len(finite_elements)
        size = nel*64
        kg_vals = np.zeros(size)
        kg_rows = np.zeros(size)
        kg_cols = np.zeros(size)

        elem = np.arange(nel)
        start = 0
        finish = 64
        for el in np.nditer(elem):
            dofs = finite_elements[el].nodal_dofs
            element_displacements = nodal_displacements[dofs]
            #stress = element.get_stresses(constitutive_model, element_displacements)
            kge_vals, kge_rows, kge_cols = finite_elements[el].get_geometric_stiffness_matrix(constitutive_model, element_displacements)
            kg_vals[start:finish] = kge_vals*densities[el]
            kg_rows[start:finish] = kge_rows
            kg_cols[start:finish] = kge_cols

            start += 64
            finish += 64

        # kg_vals = np.array(k_vals)
        # kg_rows = np.array(k_rows)
        # kg_cols = np.array(k_cols)
        #ndof = int(max(kg_cols)+1)  # +1 to convert array max "position" to "size"

        #kg = coo_matrix((kg_vals, (kg_rows, kg_cols)), shape=(ndof, ndof)).tocsc()
        return kg_vals, kg_rows, kg_cols
        #return stress


    ########################################################################
    ##################### NEW DERIVATIVE METHODS############################
    @staticmethod
    def assemble_k_derivatives(finite_elements, constitutive_model, ndof, x_min):
        """
        Consider:
        Q4 element

        K(x) = (x_min + x(1 - x_min)) K0
        dK(x)/dx = (1 - x_min)K0
        """
        elements = finite_elements
        nel = len(elements)

        k_derivs = np.zeros((8,8,nel))
        k_deriv_dofs = np.zeros((8, nel))
            
        for el in range(nel):
            (ke, el_dofs) = elements[el].get_stiffness_matrix(constitutive_model,
                                                              derivatives=True)

            k_derivs[:,:,el] = ke *(1 - x_min)
            k_deriv_dofs[:, el] = el_dofs

        return k_derivs, k_deriv_dofs


    @staticmethod
    def assemble_kg_derivatives(finite_elements, constitutive_model, displacements):
        """
        Consider:
        Q4 element
        
        Kg(x,u) = x * Kg0
        dK(x)/dx = Kg0
        """
        elements = finite_elements
        nel = len(elements)

        kg_derivs = np.zeros((8,8,nel))
                    
        for el in range(nel):
            dofs = elements[el].nodal_dofs
            el_displacements = displacements[dofs]
            kg = elements[el].get_geometric_stiffness_matrix(constitutive_model,
                                                             el_displacements,
                                                             derivatives=True)

            kg_derivs[:,:,el] = kg
           

        return kg_derivs


    @staticmethod
    def assemble_ku_derivatives(ndof,
                                finite_elements,
                                constitutive_model,
                                displacements,
                                x_min=1e-6,
                                only_data=False):
        """
        K(x) = (x_min + x(1 - x_min)) K0
        dK(x)/dx = (1 - x_min)K0

        """
        #print("Assemble call")
        elements = finite_elements
        nel = len(finite_elements)
        
        # dku_vals = np.array([])
        # dku_rows = np.array([])
        # dku_cols = np.array([])

        size = nel*8
        dku_vals = np.zeros(size)
        dku_rows = np.zeros(size)
        dku_cols = np.zeros(size)
            
        if only_data:
            for el in range(nel):
                (ke, el_dofs) = elements[el].get_stiffness_matrix(constitutive_model,
                                                              derivatives=True)
                dofs = elements[el].nodal_dofs
                ue = displacements[dofs]
                dk_dx = (1 - x_min)*ke

                dkue = dk_dx @ ue
                              
                dku_vals = np.append(dku_vals, dkue)
        
                return dku_vals

        else:
            elem = np.arange(nel)
            start = 0
            finish = 8
            for el in np.nditer(elem):
                (ke, el_dofs) = elements[el].get_stiffness_matrix(constitutive_model,
                                                              derivatives=True)
            # ke_rows = elements[el].ke_rows
            # ke_cols = elements[el].ke_cols
                dofs = el_dofs
                ue = displacements[dofs]

                ecol = np.ones(len(dofs))*el
                erow = dofs

                dk_dx = (1 - x_min)*ke

                dkue = dk_dx @ ue
                              
                dku_vals[start:finish] = dkue
                dku_rows[start:finish] = erow
                dku_cols[start:finish] = ecol

                start += 8
                finish +=8

                # dku_vals = np.append(dku_vals, dkue)
                # dku_rows = np.append(dku_rows, erow)
                # dku_cols = np.append(dku_cols, ecol)

            dku = coo_matrix((dku_vals, (dku_rows, dku_cols)), shape=(ndof, nel))  
            return dku

   
    @staticmethod
    def prepare_kg_u_derivatives(finite_elements, constitutive_model, ndof):
        """
        Considers:
        Q4 element

        Assemble without densities
        save the flat coordinates that belong to each element

        to apply densities, take the assembled vals and multiply by the densities
        at saved positions
        [vals....] *(el by el) [densities at saved positions]
        
        """
        elements = finite_elements
        nel = len(elements)

        dkg_du = dict()
        dkg_rows = dict()
        dkg_cols = dict()
        element_map = dict()
        for ui in np.arange(ndof):
            dkg_du[ui] = np.array([])
            dkg_rows[ui] = np.array([])
            dkg_cols[ui] = np.array([])
            element_map[ui] = np.array([])

        # test_dkg_du = dkg_du
        # print(test_dkg_du.keys())

        #dkg_rows = np.array([])
        #dkg_cols = np.array([])

        #element_map = np.array([])  # Record the positions of each element contribution
        e0 = np.ones(64)
        for el in range(nel):
            dkge_vals, dkge_rows, dkge_cols = elements[el].get_geometric_matrix_u_derivs(constitutive_model)
            #print(dkge_vals)

            # Inside each element, loop for each displacement, one matrix dkg/du
            #  for each dof
            el_dofs = elements[el].nodal_dofs
            for ui in el_dofs:
                dkg_du[ui] = np.append(dkg_du[ui], dkge_vals[ui])
                dkg_rows[ui] = np.append(dkg_rows[ui], dkge_rows)
                dkg_cols[ui] = np.append(dkg_cols[ui], dkge_cols)
                element_map[ui] = np.append(element_map[ui], (e0 * el))

            

        return dkg_du, dkg_rows, dkg_cols, element_map


    @staticmethod
    def assemble_dkg_du_SIMP(dkg_du, dkg_rows, dkg_cols, element_map, densities):
        """
        Build global Matrix dkg/du for each dof,
        applying the SIMP densities
        
        Kg(x,u) = x * Kg0
        dK(x,u)/du = x * dKg/du
        """
        ndof = len(dkg_du)
        rows = dkg_rows
        cols = dkg_cols


        #dofs = np.arange(ndof)
        dkg_list = np.array([])
       # print(dkg_du[0].shape)#dkg_du = csr_matrix((int(ndof**2), ndof))
        for ui in np.arange(ndof):
            # Apply densities fro SIMP
            #length = len(dkg)
            #print(length)
            #print('dkg_du_type')
            #print('ubiannoancnanwpnawnmdcpádc')
            #print(dui.shape)
            # print(dui.shape)
            #print(type(dui[2]))
            vals = apply_densities(dkg_du[ui],
                                   len(dkg_du[ui]),
                                   element_map[ui],
                                   densities)
            
            kg = coo_matrix((vals, (rows[ui], cols[ui])), shape=(ndof, ndof))
            dkg_list = np.append(dkg_list, kg)#dkg_list.append(kg)
            # kgline = kg.reshape((int(ndof ** 2), 1))
            
            # dkg_du = hstack(dkg_du, kgline)
        #t0 = time.process_time()
        dkg_du_block_d = block_diag(dkg_list, format='csc')
        #print('Build block(%f s)' % (time.process_time()-t0))
        
        return dkg_du_block_d
        

from numba import njit
@njit(fastmath=True)
def apply_densities(values, values_length, map, densities):
    """
    Densities: one value for each element
    Map: Same length as 'values', says wich element corresponds to each value.
    Values: Array to be modified
    """
    
    new_values = np.zeros(values_length)

    #print(map.shape)
    #print(values.shape)
    # print(type(values_length))

    for n in np.arange(values_length):
        # print(values)
        # print(type(values))
        new_values[n] = values[n] * densities[int(map[n])]

    return new_values



    # @staticmethod
    # def assemble_kg_u_derivatives(finite_elements, constitutive_model, ndof):
    #     """
    #     Considers:
    #     Q4 element
        
    #     """
    #     elements = finite_elements
    #     nel = len(elements)
    #     size = nel*64

    #     elem = np.arange(nel)
    #     dofs = np.arange(ndof)
    #     start = 0
    #     finish = 64
    #     dkg_du = np.zeros((size, ndof))


    #     ## Assemble indices (loop over elements only, indices don't change for
    #     # the derivatives wrt different dofs)
    #     dkgu_rows = np.zeros(size)
    #     dkgu_cols = np.zeros(size)
    #     element_map = np.zeros(size)    # Says wich element gave the 
    #                                     # contribution to the dKg/du value

    #     for el in np.nditer(elem):

    #         dkg_due, rows, cols = finite_elements[el].get_geometric_matrix_u_derivs(constitutive_model)
    #         dkgu_rows[start:finish] = rows
    #         dkgu_cols[start:finish] = cols
    #         element_map[start:finish] = el

    #         start += 64
    #         finish += 64

    #     ## Assemble values
    #     for ui in np.nditer(dofs):

    #         dkgu_vals = np.zeros(size,8)
            
    #         for el in np.nditer(elem):
    #             el_dofs = finite_elements[el].nodal_dofs

    #             dkg_due, rows, cols = finite_elements[el].get_geometric_matrix_u_derivs(constitutive_model)
    #             dkgu_vals[start:finish] = dkg_due

    #             start += 64
    #             finish += 64
            
    #         dkg_du[:,] = dkgu_vals
        
    #     return dkg_du, dkgu_rows, dkgu_cols, element_map


    # @staticmethod
    # def build_kg_u_derivs(dkg_du, dkgu_rows, dkgu_cols, element_map, densities):
    #     """
    #     """
    #     ndof