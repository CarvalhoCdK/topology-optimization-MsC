import numpy as np
#from material import Material
from fem.material import Material   #pylint: disable=F0401
from math import sqrt
from scipy.sparse import coo_matrix


class Q4_Element(object):
    """
    """

    SquareElements = True

    gauss_points = np.array([-sqrt(3)/3, sqrt(3)/3])
    gauss_weights = np.ones(2)


    def __init__(self, nodes, coordinates):

        number_of_nodes = len(nodes)
        self.number_of_dof = 2*number_of_nodes 

        self.node_map = np.zeros((number_of_nodes,2))
        self.dof_map = np.zeros((number_of_nodes,2))

        for n in range(number_of_nodes):
            self.node_map[n, :] = coordinates[n, :]
            self.dof_map[n, :] = [2*nodes[n], 2*nodes[n] + 1]
        
        self.dof_map = np.reshape(self.dof_map, (self.number_of_dof, 1))

        (j_K_mat, i_K_mat) = np.meshgrid(self.dof_map, self.dof_map)

        self.i_K = np.reshape(i_K_mat, (self.number_of_dof**2))
        self.j_K = np.reshape(j_K_mat, (self.number_of_dof**2))

    def shape_functions(self):
        pass


    def __local_shape_derivatives(self, **coordinates):
        ksi = coordinates['ksi']
        eta = coordinates['eta']

        Dl = np.zeros((2,4))

        Dl[0,0] = - 1 + eta
        Dl[0,1] =   1 - eta
        Dl[0,2] =   1 + eta
        Dl[0,3] = - 1 - eta

        Dl[1,0] = - 1 + ksi
        Dl[1,1] = - 1 - ksi
        Dl[1,2] =   1 + ksi
        Dl[1,3] =   1 - ksi

        return Dl / 4
       


    def __get_jacobian(self):

        x0 = self.node_map[0,0]
        x1 = self.node_map[1,0]
        y0 = self.node_map[0,1]
        y2 = self.node_map[2,1]

        jacobian = np.zeros((2,2))
        jacobian[0,0] = x1 - x0
        jacobian[1,1] = y2 - y0


        jacobian *= 2

        return jacobian
    

    def __get_shape_derivatives(self, **coordinates):

        local_shape_derivatives = self.__local_shape_derivatives(**coordinates)
        jacobian = self.__get_jacobian()

        global_shape_derivatives = np.dot(np.linalg.inv(jacobian), local_shape_derivatives)

        return global_shape_derivatives


    def __get_stress_strain_matrix(self, **coordinates):

        shape_derivatives = self.__get_shape_derivatives(**coordinates)
        order = len(shape_derivatives[0, :])

        B = np.zeros((3,2*order))
        for i in range(order):
            col1 = 2 * i
            col2 = col1 + 1

            B[0,col1] = shape_derivatives[0,i]
            B[1,col2] = shape_derivatives[1,i]
            B[2,col1] = shape_derivatives[1,i]
            B[2,col2] = shape_derivatives[0,i]

        return B


    def get_stiffness_matrix(self, constitutive_matrix):

        jacobian = self.__get_jacobian()
           
        j_det = np.linalg.det(jacobian)
                   
        C = constitutive_matrix

        K = np.zeros((8,8))

        for ksi in self.gauss_points:
            for eta in self.gauss_points:

                integration_point = {'ksi': ksi, 'eta': eta}
                
                B = self.__get_stress_strain_matrix(**integration_point)

                K += np.linalg.multi_dot([np.transpose(B), C, B]) * j_det

        
        # C = self.material.get_elasticity_matrix()
        # coordinates = {'ksi':0, 'eta':0}
        # B = self.__get_stress_strain_matrix(**coordinates)
        K_sparse = np.reshape(K, (self.number_of_dof**2))

        #stiffness_matrix = coo_matrix((K_sparse, (self.i_K, self.j_K)), shape=(self.number_of_dof, self.number_of_dof))

        return K_sparse, self.i_K, self.j_K



# print('Class Test')

# nodes = np.array([0, 1, 2, 3])
# coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# test_element = Q4_Element(nodes, coordinates)

## Stress Strain Matrix Test
# coordinates = {'ksi':0, 'eta':0}
# B = test_element.get_stress_strain_matrix(**coordinates)

## Material Model Test
# material_properties = {'Ex': 1,    # GPa
#                        'Ey': 1,    # GPa
#                        'NUxy': 0.3}

# test_element.set_material(**material_properties)

# ## Stifness Matrix Initial Test
# K = test_element.get_stiffness_matrix()

# print('STIFFNESS MATRIX [K]')
# np.set_printoptions(precision=4)
# print(K.toarray())
#print(test_element.i_K)
#print(test_element.j_K)

# Check Symmetry
# r = K - np.transpose(K)
# threshold_indices = r < 1e-10
# r[threshold_indices] = 0
# print('SYMMETRY CHECK')
# print(r)

