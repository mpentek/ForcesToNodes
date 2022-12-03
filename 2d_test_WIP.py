
import numpy as np

###############
# INPUT

# dimensions
lx = 32.9
ly = 64.2

# offset of starting corner
dx0 = -18.2
dy0 = -72.0

# sampling points inside the rectangle
n_total = 10 #75

# define the input forces, moments and their point of application
input_forces_and_moments = [451.3, 321.2, 3001.3]
input_forces_geom_center =[lx/2., ly/2.]

#############################
# generate nodal coordinates

def get_random_distributed_series(start, end, n_samples):
    np.random.seed(0)
    # generate randomly spaced data
    x = np.random.random(n_samples).cumsum()
    # rescale to desired range
    x = (x - x.min()) / x.ptp()
    x = (end - start) * x + start
    return x

coords_x = get_random_distributed_series(dx0, dx0 + lx, n_total)
coords_y = get_random_distributed_series(dy0, dy0 + ly, n_total)

nodal_coordinates = []
for x, y in zip(coords_x, coords_y):
    nodal_coordinates.append([x,y])
    
n_nodes = len(nodal_coordinates)
nodal_coordinates = np.asarray(nodal_coordinates)
nodes_geom_center = [np.average(nodal_coordinates[:,0]), np.average(nodal_coordinates[:,1])]

eccentricity = [(a-b) for a,b in zip(input_forces_geom_center, nodes_geom_center)]
input_moments_mod = input_forces_and_moments[2] + eccentricity[0] *  input_forces_and_moments[1] - eccentricity[1] * input_forces_and_moments[0]

#############################
# prepare mapping coefficient matrix

def map_forces_to_nodes(nodal_coordinates, nodes_geom_center, target_resultants):
    
    n_nodes = len(nodal_coordinates)    
    
    # #find maximum distance
    # x_max = 0.0
    # y_max = 0.0

    # for i in range(0,n_nodes):
    #     x_max_new = abs(nodal_coordinates[i][0]-nodes_geom_center[0])
    #     if x_max_new > x_max:
    #         x_max = x_max_new
            
    #     y_max_new = abs(nodal_coordinates[i][1]-nodes_geom_center[1])
    #     if y_max_new > y_max:
    #         y_max = y_max_new
    
    mapping_coef_matrix = np.zeros((3,2*n_nodes))
    # fx contribution
    mapping_coef_matrix[0,0::2] = 1.0
    # fy contribution
    mapping_coef_matrix[1,1::2] = 1.0    
    # mz
    for i in range(0,n_nodes):
        # dx, dy
        dx = nodal_coordinates[i][0]-nodes_geom_center[0]
        dy = nodal_coordinates[i][1]-nodes_geom_center[1] 
        
        # mz = dx * fy - dy * fx
        # fx contribution
        mapping_coef_matrix[2,2*i+0] = -dy
        # fy contribution
        mapping_coef_matrix[2,2*i+1] = dx
    
    mcm_trp = np.transpose(mapping_coef_matrix)
    LHS = np.matmul(mcm_trp, mapping_coef_matrix)
    
    # def evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs):
    #     current_nodal_forces = np.zeros(2*n_nodes)
    #     for i in range(0,n_nodes):
    #         # fx
    #         current_nodal_forces[i*2+0] = coefs[0] * target_resultants[0] / n_nodes + coefs[1] * target_resultants[2] / 2 / n_nodes * (nodal_coordinates[i][1]-nodes_geom_center[1])/y_max 
    #         # fy
    #         current_nodal_forces[i*2+1] = coefs[2] * target_resultants[1] / n_nodes + coefs[3] * target_resultants[2] / 2 / n_nodes * (nodal_coordinates[i][0]-nodes_geom_center[0])/x_max  

    #     current_resultant = np.dot(mapping_coef_matrix, current_nodal_forces)
        
    #     return current_resultant, current_nodal_forces
    
    # def current_residual(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs):
        
    #     res, forces = evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs)
    
    #     return np.linalg.norm(res - target_resultants)
    
    # from scipy.optimize import minimize
    # from functools import partial


    # init_coef = [1.0] * 2**2
    # initial_residual = current_residual(mapping_coef_matrix, 
    #                                     n_nodes, 
    #                                     nodal_coordinates, 
    #                                     nodes_geom_center, 
    #                                     target_resultants,
    #                                     init_coef)
    # print(init_coef)
    # print(initial_residual)
    
    # # using partial to fix some parameters for the
    # optimizable_function = partial(current_residual,
    #                                 mapping_coef_matrix, 
    #                                 n_nodes, 
    #                                 nodal_coordinates, 
    #                                 nodes_geom_center,
    #                                 target_resultants)
    
    # minimization_result = minimize(optimizable_function,
    #                             init_coef,
    #                             method='nelder-mead')
    
    # final_coef = minimization_result.x
    
    # final_residual = current_residual(mapping_coef_matrix, 
    #                                 n_nodes, 
    #                                 nodal_coordinates, 
    #                                 nodes_geom_center, 
    #                                 target_resultants,
    #                                 final_coef)
    # print(final_coef)
    # print(final_residual)
    
    # res, nodal_forces = evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix, 
    #                                                               n_nodes, 
    #                                                               nodal_coordinates, 
    #                                                               nodes_geom_center, 
    #                                                               target_resultants, 
    #                                                               final_coef)
    RHS = np.dot(mcm_trp, target_resultants)
    nodal_forces = np.linalg.solve(LHS, RHS)
    
    return nodal_forces


#############################
# calculate nodal forces distributed according to the shape functions
    
nodal_forces = map_forces_to_nodes(nodal_coordinates, nodes_geom_center, input_forces_and_moments[:2] + [input_moments_mod])

#############################
# perform check by calculating the recovered resultant for forces and moments from the nodal forces

recovered_forces_and_moments = [0., 0., 0.]

for i in range(n_nodes):
    # fx, fy
    fx = nodal_forces[i*2+0]
    fy = nodal_forces[i*2+1]
    
    #dx, dy
    dx = nodal_coordinates[i][0]-input_forces_geom_center[0]
    dy = nodal_coordinates[i][1]-input_forces_geom_center[1] 
    
    #Fx, Fy
    recovered_forces_and_moments[0] += fx
    recovered_forces_and_moments[1] += fy
    
    # Mz
    recovered_forces_and_moments[2] += dx *  fy - dy * fx

#############################
# certain check prints

print("Input forces and moments: ")
print("\t" + ', '.join([str(val) for val in input_forces_and_moments]))

print("Applied at: ")
print("\t" + ', '.join([str(val) for val in input_forces_geom_center]))

print("Recovered forces and moments: ")
print("\t" + ', '.join([str(val) for val in recovered_forces_and_moments]))

print("With the geometric center of the node: ")
print("\t" + ', '.join([str(val) for val in nodes_geom_center]))

print("With the difference between input and recovered values: ")
abs_diff = [abs(a-b) for a,b in zip(input_forces_and_moments, recovered_forces_and_moments)]
print("\t" + ', '.join([str(val) for val in abs_diff]))
if max(abs_diff) > 1e-3:
    raise Exception("Error too large, check code!")

print()