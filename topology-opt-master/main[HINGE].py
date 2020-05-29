from fem.fe_model import FEA_model
import numpy as np
import time
from openmdao.devtools import iprofile

from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver, SqliteRecorder, write_xdsm


from groups.simp_group import SimpGroup
from groups.direct_stability_group import DirectStabilityGroup
from groups.no_buckling_group import NoBucklingGroup
from tools.recording_tool import DataRecorder

#from components.penalization_comp import PenalizationComp



# Define problem domain =======================================================
fe = FEA_model()
nelx = 30
nely = 60
# Currently limit at 60000 elements

finite_elements, node_coordinates = fe.build_fe_space(nelx, nely)
ndof = 2*node_coordinates.shape[0]


# Material model ==============================================================
Ex = 1#0.2*1#e4#2e11#*0.5
Ey = 0.2*1#e4#2e11#*0.1
Nu = 0.3

fe.set_material({'Ex':Ex, 'Ey':Ey, 'NUxy':Nu})


# Loads =======================================================================
f = np.zeros((ndof,1))

known_loads = np.zeros(ndof, dtype=bool)
check = np.isclose(node_coordinates[:,0], nelx) # Distibuted left load
#known_loads[0::2] = check
#lk[1::2] = check

# Gao Test
check2 = np.isclose(node_coordinates[:,1], nely/2, atol=1.5)
#known_loads[0::2] = check * check2     # For x loads

known_loads[1::2] = check * check2    # For y loads
f[known_loads] = 5e-3

# Buckling test - distributed load (-x) on left
#P = 1#2.5e6
#f[known_loads] = -P
#f[2*nelx] = -P/2
#f[ndof-2] = -P/2

#### MBB
# P = 1
# f[ndof-1] = -P


# f*=2

# ## Patch Test Loads
# f[known_loads] = 1
# f[14] = 2
# f[22] = 2

fe.set_loads(f)


# Boundary conditions =========================================================
know_dofs = np.zeros(ndof, dtype=bool)
check = np.isclose(node_coordinates[:,0], 0)

## BUCKLING TEST BCs
#know_dofs[0::2] = check     # x
#know_dofs[1::2] = check     # y

## PATCH TEST BCs
know_dofs[0::2] = check
know_dofs[1::2] = check
#y_lock = [1] #,15,17]#[25, 27, 29, 31]
#know_dofs[y_lock] = True

fe.set_boundary_conditions(know_dofs)


# Check FEA setup
t0 = time.process_time()
fe.assemble_K()
print('K assembling time(%f s)' % (time.process_time()-t0))

t0 = time.process_time()
u = fe.solve_elastic()
print('Static time(%f s)' % (time.process_time()-t0))

print('Finish static')

## Solve stability
t0 = time.process_time()
fe.assemble_kg(u, np.ones(nelx*nely))
print('Kg assembling time(%f s)' % (time.process_time()-t0))

t0 = time.process_time()
eig, vec = fe.solve_stability(eigenvectors=True)
print('Buckling time(%f s)' % (time.process_time()-t0))


comp = u @ fe.f



t0 = time.process_time()
dkg_du, dkg_rows, dkg_cols, element_map = fe.prepare_kg_u_derivatives()
print('dKg/du prepare time(%f s)' % (time.process_time()-t0))


t0 = time.process_time()
dkg_du_block = fe.assemble_dkg_du_SIMP(dkg_du, dkg_rows, dkg_cols, element_map, np.ones(nelx*nely))
print('dKg/du assembling time(%f s)' % (time.process_time()-t0))

# Create Optimization Group ####################################################
penal_factor = 3.0#484
volume_fraction = 0.3
filter_radius = 2.0
buckling_load = 2.00*0.1253#*0.25#57.74#27.655#22.124#16.593
                    ##Bases_Column [0.053,  0.080, 0.106, 0.133]
                    ## Bases_Hinge [5.519, 8.278, 11.038, 13.797]


xt = np.loadtxt('nPlate_60x30_B0_densities.dat')#0.4
initial_densities = xt

## DATA RECORDER
data_recorder = DataRecorder('fPlate_60x30_B3.2[Pc=2.00]',
                            nelx=nelx,
                            nely=nely,
                            max_buckling=buckling_load,
                            penalization=penal_factor,
                            initial_densities=initial_densities,
                            material={'Ex':Ex, 'Ey':Ey, 'NUxy':Nu},
                            loads={'Val':0.005, 'direction':'x', 'dof':np.nonzero(f)[0]})


model = DirectStabilityGroup (fe_model=fe,
                              num_ele_x=nelx,
                              num_ele_y=nely,
                              penal_factor=penal_factor,
                              volume_fraction=volume_fraction,
                              filter_radius=filter_radius,
                              max_buckling_load=buckling_load,
                              initial_densities=initial_densities,
                              data_recorder=data_recorder)


model3 = NoBucklingGroup(fe_model=fe,
                        num_ele_x=nelx,
                        num_ele_y=nely,
                        penal_factor=penal_factor,
                        volume_fraction=volume_fraction,
                        filter_radius=filter_radius)

# Optimizer Setup
prob = Problem(model)

prob.driver = ScipyOptimizeDriver()
#prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-3
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 150



# Check Setup
prob.setup()
#prob.check_partials(includes='filter_comp')
#prob.setup(check=True, mode='rev')
t0 = time.process_time()
prob.run_model()
print('run_model Time(%f s)' % (time.process_time()-t0))
#prob.check_partials(includes='buckling_comp')
#prob.check_totals()

#t0 = time.process_time()
#prob.check_partials(includes='states_comp')
#print('Partials time(%f s)' % (time.process_time()-t0))
#prob.run()

topt = time.process_time()
prob.driver.options['debug_print'] = ['objs','ln_cons', 'nl_cons']
# # # #prob.driver.options['debug_print'] = ['totals']
# # # #prob.driver.options['dynamic_simul_derivs'] = True
# # # #prob.driver.options['dynamic_simul_derivs_repeat'] = 5

prob.run_driver()


# # # # # openmdao iprof_totals iprof.0
print('Optimization Time(%f s)' % (time.process_time()-topt))


x = prob['input_comp.densities']
data_recorder.get_densities(x)
data_recorder.save_data()

# ## PLOT
import matplotlib.pyplot as plt

xp = x.reshape((nely, nelx))#np.flip(np.transpose(x.reshape((nely, nelx))))
plt.imshow(-xp)
plt.set_cmap('gray')
# plt.xlabel('(a)', labelpad=10, fontsize=14)
# plt.xticks(np.arange(0, 40+1, step=40/4))
# plt.yticks(np.arange(0, 80+1, step=80/8))
plt.show()

#np.savetxt('GAO 80x40 l=-1.0.txt', x)

print('EoO')

# import numpy as np
# xt = np.loadtxt('new_results_txt\APlaca\VPlate_60x30_VAR_A4_densities.dat')
# import matplotlib.pyplot as plt

# xp = xt.reshape((60, 30))
# plt.imshow(-xp)
# plt.set_cmap('gray')
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# xt = np.loadtxt('DESP_Hinge_A4 20x40[p=3, buckling=0, vol=0.4]_densities.dat')
# xp = xt.reshape((50, 50))
# plt.imshow(-xp)
# plt.set_cmap('gray')

# plt.show()

#xt = np.loadtxt('new_results_txt\CColumnBuckling\C3_Column 60x30[p=3, buckling=0.106, vol=0.3]_densities.dat')
#B4_Column 60x30[p=3, buckling=0.133, vol=0.3
#Revisiting 70x30[buckling=0.0]_densities.dat
#prob.run_model()

# bk = ~fe.bu
# ndof = fe.ndof
# found = np.arange(ndof)
# rows = []
# cols = []
# empty = 0
# empty2 = 0
# i = 0
# for dof in bk:
#     empty += ndof*dof
#     #cols2 = np.hstack([cols2, (found + empty2)])
#     #empty2 += ndof*dof
#     if not dof:
#         cols = np.hstack([cols, (found + empty)])
#         rows = np.hstack([rows, np.ones(ndof)*i])
#         i += 1
#         empty += ndof





# # Solve linear elastic
# t0 = time.process_time()
# fe.assemble_K()
# u = fe.solve_elastic()
# print('Solve elastic time(%f s)' % (time.process_time()-t0))

# # Solve stability
# t0 = time.process_time()
# fe.assemble_kg(u)
# print('Kg assembling time(%f s)' % (time.process_time()-t0))

# t0 = time.process_time()
# eig = fe.solve_stability()
# print('Buckling time(%f s)' % (time.process_time()-t0))


# np.set_printoptions(precision=4)
# print(eig)

# print('lambda_1')
# np.set_printoptions(precision=4)
# print("{0:0.4f}".format(min(abs(eig))))


# print("EOF")

# from openmdao.api import view_model

# 
