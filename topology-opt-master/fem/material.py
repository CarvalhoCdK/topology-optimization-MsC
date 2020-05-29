import numpy as np 


class Material(object):
    """
    """

    ## Test for storing stifness matrix of regular meshes
    class_atribute = None

    @classmethod
    def class_change(cls, k):

        cls.class_atribute = k


    def __init__(self, **properties):
        self.Ex = properties['Ex']
        self.Ey = properties['Ey']
        self.NUxy = properties['NUxy']
        
        self.NUyx = self.Ey * self.NUxy / self.Ex
        self.Gxy = self.Ex / 2 / (1 + self.NUxy)


    def get_elasticity_matrix(self):
        aux1 = 1 - self.NUxy * self.NUyx

        C = np.zeros((3,3))
        C[0,0] = self.Ex
        C[0,1] = self.NUyx * self.Ex
        C[1,0] = self.NUyx * self.Ex
        C[1,1] = self.Ey
        C[2,2] = aux1 * self.Gxy

        C *= 1 / aux1

        return C


# print('Class Test')


# material_properties = {'Ex': 1,    # GPa
#                        'Ey': 1,    # GPa
#                        'NUxy': 0.3}  

# pla = Material(**material_properties)

# C = pla.get_elasticity_matrix()


# print('EoT')



