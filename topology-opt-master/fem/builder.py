from fem.mesh import Mesh   #pylint: disable=F0401
from fem.element import Q4_Element  #pylint: disable=F0401
from fem.optmized_Q4 import OptElementQ4
import numpy as np


class Builder(object):
    """
    """
    

    def __init__(self):
        self.fe_space = []


    def set_properties(self, **properties):
        self.nelx = properties['Nelx']
        self.nely = properties['Nely']


    def build_fe_space(self):

        # Build Mesh
        params = {'Number x elements': self.nelx,
                  'Number y elements': self.nely,
                  'x min': 0,
                  'y min': 0,
                  'x max': self.nelx,
                  'y max': self.nely}
        mesh = Mesh(**params)
        mesh_cells = mesh.get_elements()
        mesh_coordinates = mesh.get_coordinates()

        # Build node list
        self.node_coordinates = mesh_coordinates

        # Build Elements over the mesh
        for cell in mesh_cells:
            #self.fe_space.append(Q4_Element(cell, mesh_coordinates[cell, :]))
            self.fe_space.append(OptElementQ4(cell))

     
    def retrieve_elements(self):

        return self.fe_space


    def retrieve_nodes(self):

        return self.node_coordinates                    



## CLASS TEST
# fe_builder = Builder()
# properties = {'Nelx':10, 'Nely':10}
# fe_builder.set_properties(**properties)
# fe_builder.build_fe_space()

# print('Builder test')
# print('EoT')
## END OF CLASS TEST
