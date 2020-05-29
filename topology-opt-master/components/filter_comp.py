import numpy as np
import scipy.sparse 

from openmdao.api import ExplicitComponent


class DensityFilterComp(ExplicitComponent):
    # this simplest density filter uses <8 elements (nearest neighbors)
    def initialize(self):
        #self.options.declare('length_x', types=(int, float), )#required=True)
        #self.options.declare('length_y', types=(int, float), )#required=True)
        self.options.declare('num_ele_x', types=int)#required=True)
        self.options.declare('num_ele_y', types=int)#required=True)
        #self.options.declare('num_dvs', types=int, )#required=True)
        self.options.declare('radius', types=float)#required=True)
        
    def setup(self):
        nelx = self.options['num_ele_x']
        nely = self.options['num_ele_y']
        num_ele = nelx*nely
        radius = self.options['radius']

        self.add_input('densities', shape=num_ele)
        self.add_output('densities_f', shape=num_ele)

        lx = 1
        ly = 1

        cnt = 0

        row = []
        col = []
        wij = []
        
        cnt = 0
        ''' NOTE: element is ordered by x-y sequence '''
        for ix in range(0, nelx):
            for iy in range(0, nely):
                # eid_i  = ix * num_elem_y + iy
                eid_i = iy * nelx + ix
                # centers
                x_i = (ix + 0.5) * lx
                y_i = (iy + 0.5) * ly
                dsum = 0
                dij_list = []
                negh_list = []
                num_neighbor = 0

                for jx in range(max(0, ix - 3), min(nelx, ix + 3)):
                    for jy in range(max(0, iy - 3), min(nely, iy + 3)):
                        # eid_j  = jx * num_elem_y + jy
                        eid_j = jy * nelx + jx
                        x_j = (jx + 0.5) * lx
                        y_j = (jy + 0.5) * ly

                        dij = ((x_i-x_j)**2 + (y_i-y_j)**2)**0.5
                        # print(eid_i, eid_j, x_i, y_i, x_j, y_j, dij)
                        if dij < radius:
                            num_neighbor += 1
                            dij_list.append(dij)
                            negh_list.append(eid_j)
                            dsum += dij
                
                for tt in range(0, num_neighbor):
                    row.append(eid_i)
                    col.append(negh_list[tt])
                    wij.append(dij_list[tt]/dsum)
                    cnt += 1

        row = np.array(row)
        col = np.array(col)
        wij = np.array(wij)
        self.mtx = scipy.sparse.csr_matrix((wij, (row, col)), shape=(num_ele, num_ele))
                
        self.declare_partials('densities_f', 'densities', rows = row, cols = col, val = wij)

    def compute(self, inputs, outputs):
        outputs['densities_f'] = self.mtx.dot(inputs['densities'])