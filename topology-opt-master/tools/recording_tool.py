import numpy as np
import matplotlib.pyplot as plt


class DataRecorder(object):
    """
    Objectives 'Compliance'
    Volume Constraints
    Buckling Load Constraint
    """

    def __init__(self, process_identification:str,
                       nelx:int,
                       nely:int,
                       max_buckling,
                       penalization,
                       initial_densities,
                       material:dict={'E1':0, 'E2':0, 'nu':0},
                       loads:dict={'Val':0, 'direction':0, 'dof':0},
                       boundary='Left Clamped'):
        """
        """
        self.identification = process_identification

        self.info = ('OPTIMIZATION PARAMETERS \n'  \
                     'nelx: ' + str(nelx) + '\n'  \
                     'nely: ' + str(nely) + '\n'  \
                     'max buckling: ' + str(max_buckling) + '\n'  \
                     'penalization: ' + str(penalization) + '\n'  \
                     'initial densities: ' + str(initial_densities) + '\n'  \
                     'FEA MODEL DATA \n'  \
                     'material: ' + str(material) + '\n'  \
                     'loads: ' + str(loads) + '\n'  \
                     'bc: ' + str(boundary) + '\n')

        # Iteration number compensated for 2 initial calls to model at start
        # of openMDAO driver operation
        self.it = -1
        self.buckling_constraints = np.zeros(5)
        self.volume = np.zeros(1)
        self.compliance = np.zeros(1)

    
    ## Recording Methods
    def get_buckling_constraints(self, it_buckling_loads):
        if self.it > 0:
            self.buckling_constraints = np.vstack((self.buckling_constraints,
                                                  it_buckling_loads))
        
        self.it += 1


    def get_volume(self, it_volume):
        if self.it > 0:
            self.volume = np.vstack((self.volume,
                                     it_volume))


    def get_compliance(self, it_compliance):
        if self.it > 0:
            self.compliance = np.vstack((self.compliance,
                                         it_compliance))


    def get_densities(self, densities):
        self.densities = densities


    ## Saving methods
    def save_data(self):
        """
        it   |   compliance   |   volume    |  buckling_factors

        """

        it = np.arange(self.it, dtype=int)[1:, np.newaxis]
        buckling = self.buckling_constraints[1:,:]
        volume = self.volume[1:]
        compliance = self.compliance[1:]

        data = np.hstack((it, buckling, volume, compliance))

        header = "it, buckling[l1, l2, l3, l4, l5], volume, compliance"
        np.savetxt((self.identification + '_OptProcess' + '.dat'),
                    data,
                    header=header,
                    footer=self.info,
                    fmt='%1.4f')

        np.savetxt((self.identification + '_densities' + '.dat'),
                    self.densities,
                    header='densities',
                    footer=self.info,
                    fmt='%1.4f')

        print('Data saved')
    
    ## Ploting Methods
    # @staticmethod
    # def draw_results(identification):
    #     """
    #     """
    #     x = np.loadtxt((identification + '_densities' + '.dat'))
    #     xp = x.reshape((nely, nelx))
    #     plt.imshow(-xp)
    #     plt.set_cmap('gray')

    #     # Config
    #     #plt.title('', fontdict=font)
    #     #plt.text('', fontdict=font)
    #     plt.xlabel('time (s)', fontdict=font)
    #     plt.ylabel('voltage (mV)', fontdict=font)

    #     plt.show()







    def read_data(self):
        pass
    