from fem.fe_model import FEA_model
import numpy as np
import time
from openmdao.devtools import iprofile

from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver, SqliteRecorder


from groups.simp_group import SimpGroup
from groups.direct_stability_group import DirectStabilityGroup
from groups.no_buckling_group import NoBucklingGroup
from tools.recording_tool import DataRecorder

#from components.penalization_comp import PenalizationComp



# Define problem domain =======================================================
fe = FEA_model()
nelx = 10
nely = 10
# Currently limit at 60000 elements (yet unknown cause)

finite_elements, node_coordinates = fe.build_fe_space(nelx, nely)
ndof = 2*node_coordinates.shape[0]


# Material model ==============================================================
Ex = 0.2*1e5#2e11#*0.5
Ey = 1e5#2e11#*0.1
Nu = 0.3

fe.set_material({'Ex':Ex, 'Ey':Ey, 'NUxy':Nu})


# Loads =======================================================================
f = np.zeros((ndof,1))

known_loads = np.zeros(ndof, dtype=bool)
check = np.isclose(node_coordinates[:,0], nelx) # Distibuted left load
#known_loads[0::2] = check
#lk[1::2] = check

# Gao Test
check2 = np.isclose(node_coordinates[:,1], nely/2)
#known_loads[0::2] = check * check2     # For x loads

known_loads[1::2] = check * check2    # For y loads
f[known_loads] = -10

# Buckling test - distributed load (-x) on left
P = 1#2.5e6
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

################################################################################
################################################################################
################################################################################
################################################################################
## Experiment 1
# Create Optimization Group ####################################################
penal_factor = 3.0
volume_fraction = 0.5
filter_radius = 2.0
buckling_load = 0.5
initial_densities = 0.5

name = ('ColumnTest_[BucklingLoad: ' + str(buckling_load) + ']')

## DATA RECORDER
data_recorder1 = DataRecorder(name,
                            nelx=nelx,
                            nely=nely,
                            max_buckling=buckling_load,
                            penalization=penal_factor,
                            initial_densities=initial_densities,
                            material={'Ex':Ex, 'Ey':Ey, 'NUxy':Nu},
                            loads={'Val':-10, 'direction':'y', 'dof':np.nonzero(f)[0]})


model1 = DirectStabilityGroup (fe_model=fe,
                            num_ele_x=nelx,
                            num_ele_y=nely,
                            penal_factor=penal_factor,
                            volume_fraction=volume_fraction,
                            filter_radius=filter_radius,
                            max_buckling_load=buckling_load,
                            initial_densities=initial_densities,
                            data_recorder=data_recorder1)


# Optimizer Setup
prob = Problem(model1)

prob.driver = ScipyOptimizeDriver()
#prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-2
prob.driver.options['disp'] = True



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
#prob.driver.options['debug_print'] = ['objs','ln_cons', 'nl_cons']
# # # #prob.driver.options['debug_print'] = ['totals']
# # # #prob.driver.options['dynamic_simul_derivs'] = True
# # # #prob.driver.options['dynamic_simul_derivs_repeat'] = 5


prob.run_driver()



# # # # # openmdao iprof_totals iprof.0
print('Optimization Time(%f s)' % (time.process_time()-topt))

## SAVE DATA
x = prob['input_comp.densities']
data_recorder1.get_densities(x)
data_recorder1.save_data()
print('Done load: ' + str(buckling_load))
print(data_recorder1.identification)

################################################################################
################################################################################
################################################################################
################################################################################
## Experiment 2
# Create Optimization Group ####################################################
penal_factor = 3.0
volume_fraction = 0.5
filter_radius = 2.0
buckling_load = 0.75
initial_densities = 0.5

name = ('ColumnTest_[BucklingLoad: ' + str(buckling_load) + ']')

## DATA RECORDER
data_recorder2 = DataRecorder(name,
                            nelx=nelx,
                            nely=nely,
                            max_buckling=buckling_load,
                            penalization=penal_factor,
                            initial_densities=initial_densities,
                            material={'Ex':Ex, 'Ey':Ey, 'NUxy':Nu},
                            loads={'Val':-10, 'direction':'y', 'dof':np.nonzero(f)[0]})


model2 = DirectStabilityGroup (fe_model=fe,
                            num_ele_x=nelx,
                            num_ele_y=nely,
                            penal_factor=penal_factor,
                            volume_fraction=volume_fraction,
                            filter_radius=filter_radius,
                            max_buckling_load=buckling_load,
                            initial_densities=initial_densities,
                            data_recorder=data_recorder2)


# Optimizer Setup
prob = Problem(model2)

prob.driver = ScipyOptimizeDriver()
#prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-2
prob.driver.options['disp'] = True



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
#prob.driver.options['debug_print'] = ['objs','ln_cons', 'nl_cons']
# # # #prob.driver.options['debug_print'] = ['totals']
# # # #prob.driver.options['dynamic_simul_derivs'] = True
# # # #prob.driver.options['dynamic_simul_derivs_repeat'] = 5


prob.run_driver()



# # # # # openmdao iprof_totals iprof.0
print('Optimization Time(%f s)' % (time.process_time()-topt))

## SAVE DATA
x = prob['input_comp.densities']
data_recorder2.get_densities(x)
data_recorder2.save_data()
print('Done load: ' + str(buckling_load))
print(data_recorder2.identification)



# ## RUN EXPERIMENT
# buckling_experiment = np.array([0.5, 0.75])#, 1.0, 1.5])
# for load in buckling_experiment:
#     experiment_buckling_loads(load, fe, nelx, nely)


print('EoO')
