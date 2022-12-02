
import numpy as np

###############
# INPUT

# dimensions
lx = 32.9
ly = 64.2
lz = 73.5

# offset of starting corner
dx0 = -18.2
dy0 = -72.0
dz0 = 23.3

# sampling points inside the rectangle
nx = 7
ny = 9
nz = 11
n_total = max(nx, ny, nz)**3

# define the input forces, moments and their point of application
input_forces_and_moments = [451.3, 321.2, -512.7,-1545.2, -2117.8, 3001.3]
input_forces_geom_center =[lx/2., ly/2., lz/2.]

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
coords_z = get_random_distributed_series(dz0, dz0 + lz, n_total)

nodal_coordinates = []
for x,y,z in zip(coords_x,coords_y, coords_z):
    nodal_coordinates.append([x,y,z]) 
    
n_nodes = len(nodal_coordinates)
nodal_coordinates = np.asarray(nodal_coordinates)
nodes_geom_center = [np.average(coords_x), np.average(coords_y),np.average(coords_z)]

#############################
# prepare mapping coefficient matrix

def map_forces_to_nodes(nodal_coordinates, nodes_geom_center, target_resultants):
    
    n_nodes = len(nodal_coordinates)    
    
    #find maximum distance
    x_max = 0.0
    y_max = 0.0
    z_max = 0.0

    for i in range(0,n_nodes):
        x_max_new = abs(nodal_coordinates[i][0]-nodes_geom_center[0])
        if x_max_new > x_max:
            x_max = x_max_new
            
        y_max_new = abs(nodal_coordinates[i][1]-nodes_geom_center[1])
        if y_max_new > y_max:
            y_max = y_max_new
        
        z_max_new = abs(nodal_coordinates[i][2]-nodes_geom_center[2])
        if z_max_new > z_max:
            z_max = z_max_new
    
    mapping_coef_matrix = np.zeros((6,3*n_nodes))
    # fx
    mapping_coef_matrix[0,0::3] = 1.0
    # fy
    mapping_coef_matrix[1,1::3] = 1.0
    # fz
    mapping_coef_matrix[2,2::3] = 1.0     
    # mx, my, mz
    for i in range(0,n_nodes):
        # mx = dy * fz - dz * fy
        # fy contribution
        mapping_coef_matrix[3,3*i+1] = -(nodal_coordinates[i][2]-nodes_geom_center[2])
        # fz contribution
        mapping_coef_matrix[3,3*i+2] = (nodal_coordinates[i][1]-nodes_geom_center[1]) 
        
        # my = dz * fx - dx * fz
        # fx contribution
        mapping_coef_matrix[4,3*i+0] = (nodal_coordinates[i][2]-nodes_geom_center[2])
        # fz contribution
        mapping_coef_matrix[4,3*i+2] = -(nodal_coordinates[i][0]-nodes_geom_center[0]) 
        
        # mz = dx * fy - dy * fx
        # fx contribution
        mapping_coef_matrix[5,3*i+0] = -(nodal_coordinates[i][1]-nodes_geom_center[1])
        # fy contribution
        mapping_coef_matrix[5,3*i+1] = (nodal_coordinates[i][0]-nodes_geom_center[0]) 
    
    def evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs):
        current_nodal_forces = np.zeros(3*n_nodes)
        for i in range(0,n_nodes):
            # fx
            # to Fx
            current_nodal_forces[i*3+0] = target_resultants[0] / n_nodes 
            # to Mz
            current_nodal_forces[i*3+0] += coefs[0] * target_resultants[5] / 2 / n_nodes * (nodal_coordinates[i][1]-nodes_geom_center[1])/y_max 
            # to My
            current_nodal_forces[i*3+0] += coefs[1] * target_resultants[4] / 2 / n_nodes * (nodal_coordinates[i][2]-nodes_geom_center[2])/z_max 
            
            # fy
            # to Fy
            current_nodal_forces[i*3+1] = target_resultants[1] / n_nodes 
            # to Mz
            current_nodal_forces[i*3+1] += coefs[2] * target_resultants[5] / 2 / n_nodes * (nodal_coordinates[i][0]-nodes_geom_center[0])/x_max  
            # to Mx
            current_nodal_forces[i*3+1] += coefs[3] * target_resultants[3] / 2 / n_nodes * (nodal_coordinates[i][2]-nodes_geom_center[2])/z_max  
            
            # fz
            # to Fz
            current_nodal_forces[i*3+2] = target_resultants[2] / n_nodes 
            # to Mx
            current_nodal_forces[i*3+2] += coefs[4] * target_resultants[3] / 2 / n_nodes * (nodal_coordinates[i][1]-nodes_geom_center[1])/y_max  
            # to My
            current_nodal_forces[i*3+2] += coefs[5] * target_resultants[4] / 2 / n_nodes * (nodal_coordinates[i][0]-nodes_geom_center[0])/x_max  
            
        current_resultant = np.dot(mapping_coef_matrix, current_nodal_forces)
        
        return current_resultant, current_nodal_forces
    
    def current_residual(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs):
        
        res, forces = evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs)
    
        my_res = np.mean([(a-b)**2/b**2 for a,b in zip(res, target_resultants)])
    
        return my_res
    
    from scipy.optimize import minimize
    from functools import partial


    init_coef = [1.0] * 6 #* 3**2
    initial_residual = current_residual(mapping_coef_matrix, 
                                        n_nodes, 
                                        nodal_coordinates, 
                                        nodes_geom_center, 
                                        target_resultants,
                                        init_coef)
    print(init_coef)
    print(initial_residual)
    
    # using partial to fix some parameters for the
    optimizable_function = partial(current_residual,
                                    mapping_coef_matrix, 
                                    n_nodes, 
                                    nodal_coordinates, 
                                    nodes_geom_center,
                                    target_resultants)
    
    minimization_result = minimize(optimizable_function,
                                init_coef,
                                method='Nelder-Mead')
    # ,
    #                             options={'disp': 1, 
    #                                      'maxcor': 10, 
    #                                      'ftol': 2.220446049250313e-09, 
    #                                      'gtol': 1e-05, 
    #                                      'eps': 1e-08, 
    #                                      'maxfun': 15000, 
    #                                      'maxiter': 15000, 
    #                                      'iprint': - 1, 
    #                                      'maxls': 20, 
    #                                      'finite_diff_rel_step': None})
    
    final_coef = minimization_result.x
    
    final_residual = current_residual(mapping_coef_matrix, 
                                    n_nodes, 
                                    nodal_coordinates, 
                                    nodes_geom_center, 
                                    target_resultants,
                                    final_coef)
    print(final_coef)
    print(final_residual)
    
    res, nodal_forces = evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix, 
                                                                  n_nodes, 
                                                                  nodal_coordinates, 
                                                                  nodes_geom_center, 
                                                                  target_resultants, 
                                                                  final_coef)
    
    return nodal_forces


#############################
# calculate nodal forces distributed according to the shape functions
    
nodal_forces = map_forces_to_nodes(nodal_coordinates, input_forces_geom_center, input_forces_and_moments)

#############################
# perform check by calculating the recovered resultant for forces and moments from the nodal forces

recovered_forces_and_moments = [0., 0., 0., 0., 0., 0.]

for i in range(n_nodes):
    # Fx, Fy, Fz
    recovered_forces_and_moments[0] += nodal_forces[i*3+0]
    recovered_forces_and_moments[1] += nodal_forces[i*3+1]
    recovered_forces_and_moments[2] += nodal_forces[i*3+2]

    # Mx, My, Mz
    recovered_forces_and_moments[3] += (nodal_coordinates[i][1]-input_forces_geom_center[1]) *  nodal_forces[i*3+2] - (nodal_coordinates[i][2]-input_forces_geom_center[2]) * nodal_forces[i*3+1]
    recovered_forces_and_moments[4] += (nodal_coordinates[i][2]-input_forces_geom_center[2]) *  nodal_forces[i*3+0] - (nodal_coordinates[i][0]-input_forces_geom_center[0]) * nodal_forces[i*3+2]
    recovered_forces_and_moments[5] += (nodal_coordinates[i][0]-input_forces_geom_center[0]) *  nodal_forces[i*3+1] - (nodal_coordinates[i][1]-input_forces_geom_center[1]) * nodal_forces[i*3+0]

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