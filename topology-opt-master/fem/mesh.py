import numpy as np
import matplotlib.pyplot as plt

class Mesh(object):
    """
    2D rectangle given an initial point (x0, y0) and an end point (x1, y1).
    Resulting in a number of elements equal to nx*ny.

    1----2----3----4------x       1 = (xo, yo)
    |    |    |    |             12 = (xf, yf)
    |    |    |    |
    5----6----7----8
    |    |    |    |
    |    |    |    |
    9---10---11---12
    |
    |
    y

    Args:
        xo (float): x min
        yo (float): y min
        x1 (float): x max
        y1 (float): x min
        n_elx (int): Number of elements in x direction
        n_ely (int): Number of elements in y direction

    """

    type = 'Rectangular'


    def __init__(self, **params):
        self.n_elx = params['Number x elements']
        self.n_ely = params['Number y elements']
        self.xo = params['x min']
        self.yo = params['y min']
        self.xf = params['x max']
        self.yf = params['y max']

        # Generate vertices coordinates
        self.vertex_coordinates = np.empty([(self.n_elx+1) * (self.n_ely +1),2])
        lx = np.linspace(self.xo, self.xf, num = self.n_elx + 1)
        ly = np.linspace(self.yo, self.yf, num = self.n_ely + 1)
        n = 0
        for y in ly:
            for x in lx:
                self.vertex_coordinates[n, 0] = x
                self.vertex_coordinates[n, 1] = y
                n += 1
            
        # Identify element vertices
        self.elements = np.empty([self.n_elx * self.n_ely, 4], dtype = np.uint16)
        for el in range(self.n_elx * self.n_ely):
            line = el // self.n_elx
            n1 = el + line
            n2 = n1 + 1
            n3 = n2 + self.n_elx
            n4 = n3 + 1
            self.elements[el,:] = [n1, n2, n3, n4]


    def get_coordinates(self):
        """
        Returns:
        array[x, y]: Each row corresponds to a vertex
        """
        return self.vertex_coordinates


    def get_elements(self):
        """
        Returns:
        array[n1, n2, n3, n4]: Each row contains the 4 vertices that build an 
        element.
        """
        return self.elements


    def plot(self):
        plt.scatter(self.vertex_coordinates[:,0], self.vertex_coordinates[:,1],
                    linewidths=1)
        plt.grid(alpha = 1)
        plt.show()



# Class tests
# print('Class tests')

# from mesh import Mesh

# params = {'Number x elements': 2,
#           'Number y elements': 2,
#           'x min': 0,
#           'y min': 0,
#           'x max': 1,
#           'y max': 2}

# mesh = Mesh(**params)
# mesh_cells = mesh.get_elements()
# mesh_coordinates = mesh.get_coordinates()

# print('EoT')