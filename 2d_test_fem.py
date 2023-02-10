
import numpy as np

#############################
# OWN function definition START


def get_random_distributed_series(start, end, n_samples, fix_seed=True):
    if fix_seed:
        np.random.seed(0)
    # generate randomly spaced data
    x = np.random.random(n_samples).cumsum()
    # rescale to desired range
    x = (x - x.min()) / x.ptp()
    x = (end - start) * x + start
    return x

def setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center):
    E = 1000.0
    A = 10.0
    I = 100.0
    
    def get_length_and_angle_coefs(p1, p2):
        # http://what-when-how.com/the-finite-element-method/fem-for-frames-finite-element-method-part-1/
        length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
        cos_val = (p2[0]-p1[0]) / length
        sin_val = (p2[1]-p1[1]) / length
        
        return length, cos_val, sin_val
    
    k_total_global =np.zeros((3 + 2*len(nodal_coordinates), 3 + 2*len(nodal_coordinates)))
    
    for idx, node in enumerate(nodal_coordinates):
        L, c_val, s_val = get_length_and_angle_coefs(nodes_geom_center, node)
        
        k_u = E*A/L 
        k_vv = 3*E*I/L**3
        k_vr = 3*E*I/L**2
        k_rr = 3*E*I/L
        
        k_elem_local = np.array([
            [ k_u,   0.0,   0.0, -k_u,    0.0],
            [ 0.0,  k_vv,   k_vr,  0.0,  -k_vv],
            [ 0.0,  k_vr,   k_rr,  0.0,  -k_vr],
            [-k_u,   0.0,   0.0,  k_u,    0.0],
            [ 0.0, -k_vv,  -k_vr,  0.0,   k_vv]])
        
        t_elem = np.array([
            [ c_val, s_val, 0.0,   0.0,    0.0],
            [ -s_val,  c_val, 0.0,   0.0,    0.0],
            [   0.0,   0.0,  1.0,   0.0,    0.0],
            [   0.0,   0.0,  0.0,  c_val, s_val],
            [   0.0,   0.0,  0.0,  -s_val,  c_val]])
        
        k_elem_global = np.matmul(np.matmul(np.transpose(t_elem), k_elem_local), t_elem)
        
        # add diagonally-clustered entries corresponding to the starting node - i.e center node - of the beam
        for i in range(3):
            for j in range(3):
                k_total_global[i,j] += k_elem_global[i,j]
                
                # print(np.array_str(k_total_global[0:10,0:10], precision=2, suppress_small=True))
        
        # add diagonally-clustered entries corresponding to the end node of of the beam
        for i in range(2):
            for j in range(2):
                k_total_global[3 + idx*2 + i, 3 + idx*2 + j] += k_elem_global[3+i,3+j]
                
                # print(np.array_str(k_total_global[0:10,0:10], precision=2, suppress_small=True))
        
        # add coupling terms between nodes, which are off-diagonal
        for i in range(2):
            for j in range(3):
                # lower diagonal
                k_total_global[3 + idx*2 + i, j] += k_elem_global[3+i,j]
                # upper diagonal
                k_total_global[j, 3 + idx*2 + i] += k_elem_global[j,3+i]
                
                # print(np.array_str(k_total_global[0:10,0:10], precision=2, suppress_small=True))
    
    return k_total_global

def map_forces_to_nodes(nodal_coordinates, nodes_geom_center, target_resultants):

    # setup a stiffness matrix assuming the center node being connected to all other nodes
    # by a beam
    stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center)
    
    # calculate the 3 deformations - 2 translations and 1 rotation - of the center node
    # using the displacement method in 2D
    # solved by reduction, only the center node is unconstrained
    center_node_deformations = np.linalg.solve(stiffness_matrix[:3,:3], target_resultants)
    
    # setup the deformation vector - apart from the center node deformations
    # all are zero translations due to the pinned support
    all_deformations = np.zeros(3+2*len(nodal_coordinates))
    # only nonzero which were previously solved for
    all_deformations[:3] = center_node_deformations
    
    # recover all forces
    all_forces = np.dot(stiffness_matrix,all_deformations)
    # first 3 are the recovered resultants
    recovered_resultants = all_forces[:3]
    # the remaining are the actual unknowns of the problem
    # return the reaction forces from each pinned node
    # flip sign for consistency
    nodal_forces = -all_forces[3:]
    
    # check target forces in the center node
    residual = np.subtract(target_resultants, recovered_resultants)
    norm_of_residual = np.linalg.norm(residual)
    print(norm_of_residual)
    if norm_of_residual > 1e-4:
        raise Exception("Norm of residual too large, check algorithm!")




    ##########################
    # check based on nodal position
    n_nodes = len(nodal_coordinates)

    mapping_coef_matrix = np.zeros((3, 2*n_nodes))

    # fx contribution
    mapping_coef_matrix[0, 0::2] = 1.0
    # fy contribution
    mapping_coef_matrix[1, 1::2] = 1.0
    # mz
    for i in range(0, n_nodes):
        # dx, dy
        dx = nodal_coordinates[i][0] - nodes_geom_center[0]
        dy = nodal_coordinates[i][1] - nodes_geom_center[1]

        # mz = dx * fy - dy * fx
        # fx contribution
        mapping_coef_matrix[2, 2*i+0] = -dy
        # fy contribution
        # NOTE:check why this needs a sign flip to be correct, a sign-flip might be needed somewhere else!!!
        mapping_coef_matrix[2, 2*i+1] = dx #dx

    recovered_resultants = np.dot(mapping_coef_matrix, nodal_forces)
    residual = np.subtract(recovered_resultants, target_resultants)
    norm_of_residual = np.linalg.norm(residual)
    if norm_of_residual > 1e-4:
        raise Exception("Norm of residual too large, check algorithm!")

    return nodal_forces

# OWN function definition END
#############################


#############################
# INPUT of setup START

# dimensions
lx = 32.9
ly = 64.2

# offset of starting corner
dx0 = -18.2
dy0 = -72.0

# sampling points inside the rectangle
n_total = 200#100

# define the input forces, moments and their point of application
# as provided by user

# create a realistic moment base on force and lever arm
applied_force = [451.3, 321.2]
applied_at_location = [-78.12, 8.33]

input_forces_geom_center = [lx/2., ly/2.]
input_forces_and_moments = []
# Fx, Fy, Fy
input_forces_and_moments.extend(applied_force)
# Mz = Dx * Fy - Dy * Fx
input_forces_and_moments.append((applied_at_location[0]-input_forces_geom_center[0])*applied_force[1] - (
    applied_at_location[1]-input_forces_geom_center[1])*applied_force[0])


#############################
# generate nodal coordinates

coords_x = get_random_distributed_series(dx0, dx0 + lx, n_total)
coords_y = get_random_distributed_series(dy0, dy0 + ly, n_total)

nodal_coordinates = []
for x, y in zip(coords_x, coords_y):
    nodal_coordinates.append([x, y])

# INPUT of setup END
#############################


#############################
# COMPUTATION of setup mapping START

n_nodes = len(nodal_coordinates)
nodal_coordinates = np.asarray(nodal_coordinates)
nodes_geom_center = [np.average(
    nodal_coordinates[:, 0]), np.average(nodal_coordinates[:, 1])]

# NOTE: might not be necessary to include eccentricity like this
eccentricity = [(a-b)
                for a, b in zip(input_forces_geom_center, nodes_geom_center)]
input_moments_mod = input_forces_and_moments[2] + eccentricity[0] * \
    input_forces_and_moments[1] - eccentricity[1] * input_forces_and_moments[0]

#############################
# calculate nodal forces distributed according to the shape functions

nodal_forces = map_forces_to_nodes(
    nodal_coordinates, nodes_geom_center, input_forces_and_moments[:2] + [input_moments_mod])

# COMPUTATION of setup mapping END
#############################


#############################
# CHECKS of correctness START

#############################
# perform check by calculating the recovered resultant for forces and moments from the nodal forces

recovered_forces_and_moments = [0., 0., 0.]

for i in range(n_nodes):
    # fx, fy
    fx = nodal_forces[i*2+0]
    fy = nodal_forces[i*2+1]

    # dx, dy
    dx = nodal_coordinates[i][0] - input_forces_geom_center[0]
    dy = nodal_coordinates[i][1] - input_forces_geom_center[1]

    # Fx, Fy
    recovered_forces_and_moments[0] += fx
    recovered_forces_and_moments[1] += fy

    # Mz
    recovered_forces_and_moments[2] += dx * fy - dy * fx

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

print("With the eccentricity of concentrated forces with respect to nodes center: ")
print("\t" + ', '.join([str(val) for val in eccentricity]))

print("With the difference between input and recovered values: ")
abs_diff = [abs(a-b) for a, b in zip(input_forces_and_moments,
                                     recovered_forces_and_moments)]
print("\t" + ', '.join([str(val) for val in abs_diff]))
if max(abs_diff) > 1e-3:
    raise Exception("Error too large, check code!")

print()
